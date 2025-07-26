import argparse
import os
from transformers import AutoTokenizer
from optimum.exporters.onnx import main_export
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort


def export_to_onnx(model_id: str, output_path: str):
    print(f"🔄 Exporting model '{model_id}' to ONNX...")
    main_export(
        model_name_or_path=model_id,
        output=output_path,
        task="text-classification",
        framework="pt"
    )
    print("✅ ONNX export complete.")


def quantize_model(onnx_input: str, onnx_output: str):
    print("🔧 Quantizing ONNX model to INT8...")
    quantize_dynamic(
        model_input=onnx_input,
        model_output=onnx_output,
        weight_type=QuantType.QInt8
    )
    print(f"✅ Quantized model saved to: {onnx_output}")


def run_inference(quantized_model_path: str, model_id: str, text: str):
    print("🤖 Running inference on quantized model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(text, return_tensors="np", padding=True)

    session = ort.InferenceSession(quantized_model_path)
    outputs = session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })
    print("🧠 Logits:", outputs[0])


def main():
    parser = argparse.ArgumentParser(description="Quantize a Hugging Face model to ONNX INT8")
    parser.add_argument("--model", required=True, help="Hugging Face model ID (e.g., bhadresh/...)")
    parser.add_argument("--output", default="./onnx_model", help="Directory to save ONNX files")
    parser.add_argument("--text", default="I love this product!", help="Test input text for inference")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    onnx_fp32_path = os.path.join(args.output, "model.onnx")
    onnx_int8_path = os.path.join(args.output, "model-quant.onnx")

    export_to_onnx(args.model, args.output)
    quantize_model(onnx_fp32_path, onnx_int8_path)
    run_inference(onnx_int8_path, args.model, args.text)


if __name__ == "__main__":
    main()
