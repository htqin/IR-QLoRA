# IR-QLoRA with Integer Quantizer



## Getting Started

This tutorial explains how to implement IR-QLoRA with Integer Quantizer.

### Get int4 model using ICQ
To implement ICQ, you need to replace the integer quantizer. To do this, simply:

- Replace `auto_gptq/quantization/quantizer.py` with our `quantizer_icq.py`
- Perform int4 quantization as demonstrated in [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)

### Fine-tune with IEC

To implement IEC, modifications need to be made to the structure of LoRA. Simply follow these steps:

- Replace `auto_gptq/utils/peft_utils.py` with our `peft_utils_iec.py`
- Perform fine-tuning as demonstrated in [QA-LoRA](https://github.com/yuhuixu1993/qa-lora)



## Acknowledgements

Our code is based on [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ), [QA-LoRA](https://github.com/yuhuixu1993/qa-lora)
