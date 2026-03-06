"""Export TableFormer to 3 ONNX models: encoder, decoder, bbox_decoder."""

from pathlib import Path

import torch

from common.encoder import TableFormerEncoder
from common.decoder import TableFormerDecoder
from common.bbox_decoder import BBoxDecoder
from common.export_common import export_encoder, export_decoder, export_bbox_decoder
from common.weights import download_weights, load_weights


def main():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Download and load weights
    print("Downloading weights...")
    safetensors_path = download_weights()
    print(f"  {safetensors_path}")

    print("Loading weights...")
    encoder_sd, decoder_sd, bbox_sd = load_weights(safetensors_path)

    # Build encoder
    print("Building encoder...")
    encoder = TableFormerEncoder()
    encoder.load_state_dict(encoder_sd)
    encoder.eval()

    # Build decoder
    print("Building decoder...")
    decoder = TableFormerDecoder()
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    # Build bbox decoder
    print("Building bbox decoder...")
    bbox_decoder = BBoxDecoder()
    bbox_decoder.load_weights(bbox_sd)
    bbox_decoder.eval()

    # Verify PyTorch forward passes
    print("Verifying PyTorch forward passes...")
    with torch.no_grad():
        dummy_img = torch.randn(1, 3, 448, 448)
        enc_mem, enc_raw = encoder(dummy_img)
        print(f"  encoder_memory: {enc_mem.shape}, enc_out_raw: {enc_raw.shape}")

        dummy_ids = torch.zeros(1, 1, dtype=torch.long)
        dummy_step = torch.zeros(1, dtype=torch.long)
        from common.weights import N_HEADS, HEAD_DIM, N_LAYERS
        from common.decoder import MAX_SEQ

        dummy_kv = [torch.zeros(1, N_HEADS, MAX_SEQ, HEAD_DIM)] * (N_LAYERS * 2)
        dec_out = decoder(dummy_ids, enc_mem, dummy_step, *dummy_kv)
        print(f"  logits: {dec_out[0].shape}, hidden_state: {dec_out[1].shape}")

        dummy_cells = torch.randn(10, 512)
        bboxes, classes = bbox_decoder(enc_raw, dummy_cells)
        print(f"  bboxes: {bboxes.shape}, classes: {classes.shape}")

    # Export
    print("Exporting ONNX models...")
    export_encoder(encoder, output_dir / "encoder.onnx")
    export_decoder(decoder, output_dir / "decoder.onnx")
    export_bbox_decoder(bbox_decoder, output_dir / "bbox_decoder.onnx")

    print("Done!")


if __name__ == "__main__":
    main()
