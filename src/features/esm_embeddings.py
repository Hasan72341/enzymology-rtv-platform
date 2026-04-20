"""
ESM-2 sequence embedding generation via local PyTorch or ONNX Runtime inference.
No external API calls. Fully offline-capable.
"""
import numpy as np
import pandas as pd
import torch
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
import hashlib

logger = logging.getLogger(__name__)


class ESM2Embedder:
    """
    Generate ESM-2 protein sequence embeddings locally.

    Supports two inference backends (controlled by agent.yaml):
      - 'pytorch'  : HuggingFace transformers + PyTorch  (GPU/CPU)
      - 'onnx'     : ONNX Runtime on CPU  (faster on CPU-only machines)

    The provider key in agent.yaml is kept as 'local' for backwards
    compatibility; internally we auto-select pytorch vs. onnx based on the
    optional 'backend' sub-key (default: 'pytorch').
    """

    # Map friendly model names → HuggingFace model IDs
    MODEL_MAP = {
        'esm2_t33_650M': 'facebook/esm2_t33_650M_UR50D',
        'esm2_t30_150M': 'facebook/esm2_t30_150M_UR50D',
        'esm2_t12_35M':  'facebook/esm2_t12_35M_UR50D',
        'esm2_t6_8M':    'facebook/esm2_t6_8M_UR50D',
    }
    
    # Map model names → hidden dimensions
    DIM_MAP = {
        'esm2_t33_650M': 1280,
        'esm2_t30_150M': 640,
        'esm2_t12_35M':  480,
        'esm2_t6_8M':    320,
    }

    def __init__(self, config: Dict, cache_dir: str = "outputs/embeddings"):
        """
        Initialise ESM-2 embedder.

        Args:
            config: feature_engineering section from agent.yaml
            cache_dir: directory for caching .npy embedding files
        """
        emb_cfg = config['sequence_embedding']
        self.model_name  = emb_cfg['model']
        self.pooling     = emb_cfg['pooling']
        
        # Auto-detect dimension if not specified or as a safety check
        self.output_dim  = self.DIM_MAP.get(self.model_name, emb_cfg.get('output_dim', 480))
        
        self.backend     = emb_cfg.get('backend', 'pytorch')   # 'pytorch' | 'onnx'
        self.batch_size  = emb_cfg.get('batch_size', 8)
        self.max_length  = emb_cfg.get('max_length', 1024)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.hf_model_name = self.MODEL_MAP.get(self.model_name, self.model_name)

        self.model     = None
        self.tokenizer = None
        self.ort_session = None

        if self.backend == 'onnx':
            self._init_onnx()
        else:
            self._init_pytorch()

    # ------------------------------------------------------------------ #
    #  Initialisation helpers                                              #
    # ------------------------------------------------------------------ #

    def _init_pytorch(self):
        """Load ESM-2 model weights via HuggingFace transformers."""
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers>=4.30.0 and torch>=2.0.0 are required. "
                "Install: pip install transformers torch"
            )

        logger.info(f"[pytorch] Loading ESM-2 model: {self.hf_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        self.model     = AutoModel.from_pretrained(self.hf_model_name)

        # Use MPS (Apple Silicon) if available, then CUDA, then CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple MPS (Metal) for inference")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA GPU for inference")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for inference")

        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("ESM-2 PyTorch model loaded ✓")

    def _init_onnx(self):
        """
        Export the ESM-2 model to ONNX (once) then load an OrtSession.
        Falls back to pytorch if onnxruntime is not installed.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning(
                "onnxruntime not found – falling back to PyTorch backend. "
                "Install: pip install onnxruntime"
            )
            self.backend = 'pytorch'
            self._init_pytorch()
            return

        onnx_path = self.cache_dir / f"{self.model_name}.onnx"

        if not onnx_path.exists():
            self._export_to_onnx(onnx_path)

        logger.info(f"[onnx] Loading ONNX session from {onnx_path}")
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        self.ort_session = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers)

        # Still need the tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        logger.info("ESM-2 ONNX session loaded ✓")

    def _export_to_onnx(self, onnx_path: Path):
        """Export the PyTorch model to ONNX format (run once, then cached)."""
        import torch.onnx
        from transformers import AutoTokenizer, AutoModel

        logger.info(f"Exporting {self.hf_model_name} → ONNX (this may take a few minutes)…")
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        model     = AutoModel.from_pretrained(self.hf_model_name)
        model.eval()

        # Create a dummy input for tracing
        dummy_seq  = ["MKTIIALSYIFCLVFA"]
        inputs     = tokenizer(dummy_seq, return_tensors='pt', padding=True,
                               truncation=True, max_length=self.max_length)

        with torch.no_grad():
            torch.onnx.export(
                model,
                args=(inputs['input_ids'], inputs['attention_mask']),
                f=str(onnx_path),
                input_names=['input_ids', 'attention_mask'],
                output_names=['last_hidden_state'],
                dynamic_axes={
                    'input_ids':       {0: 'batch', 1: 'seq_len'},
                    'attention_mask':  {0: 'batch', 1: 'seq_len'},
                    'last_hidden_state': {0: 'batch', 1: 'seq_len'},
                },
                opset_version=14,
            )

        logger.info(f"ONNX model saved to {onnx_path} ✓")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def embed_sequences(self, df: pd.DataFrame, dataset_name: str) -> np.ndarray:
        """
        Generate (and cache) embeddings for all sequences in the DataFrame.

        Args:
            df:           DataFrame containing a 'sequence' column
            dataset_name: label used in the cache filename

        Returns:
            np.ndarray of shape (n_samples, output_dim)
        """
        cache_file = self._cache_path(df, dataset_name)
        if cache_file.exists():
            logger.info(f"Loading cached embeddings from {cache_file}")
            return np.load(cache_file)

        logger.info(f"Generating {self.backend.upper()} embeddings for {len(df)} sequences…")

        sequences = df['sequence'].tolist()
        if self.backend == 'onnx' and self.ort_session is not None:
            embeddings = self._embed_onnx(sequences)
        else:
            embeddings = self._embed_pytorch(sequences)

        np.save(cache_file, embeddings)
        logger.info(f"Cached embeddings → {cache_file} ({embeddings.shape})")
        return embeddings

    # ------------------------------------------------------------------ #
    #  Backend implementations                                             #
    # ------------------------------------------------------------------ #

    def _embed_pytorch(self, sequences: List[str]) -> np.ndarray:
        """Run forward pass through the PyTorch ESM-2 model."""
        if not sequences:
            return np.empty((0, self.output_dim))

        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state  # (B, L, D)
                pooled = self._pool(hidden, inputs['attention_mask'])

            all_embeddings.append(pooled.cpu().float().numpy())

            if (i + self.batch_size) % 64 == 0 or (i + self.batch_size) >= len(sequences):
                logger.info(f"  {min(i + self.batch_size, len(sequences))}/{len(sequences)} sequences done")

        return np.vstack(all_embeddings)

    def _embed_onnx(self, sequences: List[str]) -> np.ndarray:
        """Run inference through the ONNX Runtime session (CPU-optimised)."""
        if not sequences:
            return np.empty((0, self.output_dim))

        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors='np',      # numpy tensors for ORT
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            ort_inputs = {
                'input_ids':      inputs['input_ids'].astype(np.int64),
                'attention_mask': inputs['attention_mask'].astype(np.int64),
            }

            result = self.ort_session.run(['last_hidden_state'], ort_inputs)
            hidden = result[0]                               # (B, L, D)
            mask   = inputs['attention_mask'][:, :, None]   # (B, L, 1)

            if self.pooling == 'mean':
                pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
            elif self.pooling == 'cls':
                pooled = hidden[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling strategy: '{self.pooling}'")

            all_embeddings.append(pooled.astype(np.float32))

            if (i + self.batch_size) % 64 == 0 or (i + self.batch_size) >= len(sequences):
                logger.info(f"  {min(i + self.batch_size, len(sequences))}/{len(sequences)} sequences done")

        return np.vstack(all_embeddings)

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean or CLS pooling over the sequence dimension."""
        if self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == 'cls':
            return hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: '{self.pooling}'")

    def _cache_path(self, df: pd.DataFrame, dataset_name: str) -> Path:
        """Deterministic cache filename based on a hash of the sequences."""
        seq_hash = hashlib.md5("".join(df['sequence'].tolist()).encode()).hexdigest()[:12]
        return self.cache_dir / f"{dataset_name}_{self.model_name}_{self.pooling}_{seq_hash}.npy"
