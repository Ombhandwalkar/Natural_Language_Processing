# Parameter-Efficient Fine-Tuning (PEFT) Techniques Repository

A comprehensive collection of notebooks demonstrating various Parameter-Efficient Fine-Tuning (PEFT) methods for large language models. This repository provides practical implementations of cutting-edge techniques that enable fine-tuning of large models with minimal computational resources.

## 📚 Overview

This repository contains 10 Jupyter notebooks showcasing different PEFT techniques applied to RoBERTa-large and other transformer models. These methods allow you to fine-tune large models using only a fraction of the parameters, significantly reducing memory requirements and training time while maintaining competitive performance.

## 🎯 What is PEFT?

Parameter-Efficient Fine-Tuning (PEFT) methods enable adaptation of pre-trained large language models to specific tasks without fine-tuning all model parameters. Instead of updating billions of parameters, PEFT techniques modify only a small subset, making it:

- **Memory Efficient**: Requires significantly less GPU memory
- **Cost-Effective**: Reduces computational costs dramatically
- **Fast**: Shorter training times compared to full fine-tuning
- **Practical**: Makes fine-tuning large models accessible on consumer hardware

## 📂 Repository Contents

### Sequence Classification Notebooks

1. **LoRA_roberta_large_peft_Seq_classification.ipynb**
   - Low-Rank Adaptation (LoRA) for sequence classification
   - Injects trainable rank decomposition matrices into transformer layers
   - Highly efficient and widely adopted technique

2. **IA3_roberta_large_peft_Seq_Classification.ipynb**
   - Infused Adapter by Inhibiting and Amplifying Inner Activations (IA³)
   - Rescales inner activations with learned vectors
   - Even more parameter-efficient than LoRA

3. **Prefix_tuning_roberta_large_Seq_classification.ipynb**
   - Prefix Tuning for sequence classification tasks
   - Prepends trainable continuous vectors to each layer
   - Effective for various NLP tasks

4. **Prompt_Tuning_roberta_large_Seq_Classification.ipynb**
   - Soft prompt tuning implementation
   - Learns continuous prompt embeddings
   - Simplest form of PEFT

5. **P_Tuning_roberta_large_Seq_Classification.ipynb**
   - P-Tuning v2 implementation
   - Continuous prompt tuning with deep prompt encoding
   - Bridges gap between prompt and prefix tuning

### Sequence-to-Sequence Notebooks

6. **peft_lora_seq2seq.ipynb**
   - LoRA applied to sequence-to-sequence models
   - Demonstrates LoRA for generation tasks

7. **peft_ia3_seq2seq.ipynb**
   - IA³ for sequence-to-sequence tasks
   - Shows IA³ effectiveness in generative scenarios

### Causal Language Modeling Notebooks

8. **prefix_tuning_clm.ipynb**
   - Prefix tuning for causal language modeling
   - Text generation with efficient fine-tuning

9. **lora_clm_accelerate_big_model_inference.ipynb**
   - LoRA with Accelerate library for large model inference
   - Demonstrates efficient inference with quantization and acceleration

10. **lora_clm_with_additional_tokens.ipynb**
    - LoRA with custom vocabulary expansion
    - Shows how to add new tokens while using PEFT

### Model Artifacts

- **adapter_config.json**: Configuration file for PEFT adapters
- **adapter_model.safetensors**: Saved adapter weights (SafeTensors format)

## 🚀 Getting Started

### Prerequisites

```bash
pip install transformers
pip install peft
pip install datasets
pip install accelerate
pip install torch
pip install safetensors
```

### Quick Start

1. Clone the repository
2. Install the required dependencies
3. Open any notebook in Jupyter/Colab
4. Follow the step-by-step implementation

Each notebook is self-contained with detailed explanations and can be run independently.

## 💡 Key Techniques Covered

| Technique | Parameters Modified | Best Use Case | Memory Savings |
|-----------|-------------------|---------------|----------------|
| **LoRA** | ~0.1-1% | General purpose, highly effective | High |
| **IA³** | ~0.01% | Maximum efficiency needed | Very High |
| **Prefix Tuning** | ~0.1-3% | Generation tasks | High |
| **Prompt Tuning** | ~0.01-0.1% | Simple adaptation | Very High |
| **P-Tuning** | ~0.1-1% | Balance of performance and efficiency | High |

## 🎓 Learning Path

**Beginners**: Start with `Prompt_Tuning` → `LoRA` → `Prefix_Tuning`

**Intermediate**: Explore `IA3` → `P_Tuning` → Seq2Seq implementations

**Advanced**: Dive into `lora_clm_accelerate_big_model_inference` for production scenarios

## 🔧 Use Cases

- Fine-tuning large models on custom datasets
- Adapting models for domain-specific tasks
- Multi-task learning with parameter isolation
- Rapid prototyping and experimentation
- Production deployment with limited resources

## 📊 Benefits

✅ Train large models on consumer GPUs (even single GPU)  
✅ Reduce training time by 2-10x  
✅ Store multiple task-specific adapters (MBs instead of GBs)  
✅ Easy switching between different fine-tuned versions  
✅ Maintain base model performance while specializing  

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new PEFT techniques
- Improve existing notebooks
- Share results and benchmarks
- Report issues or suggest enhancements

## 📖 Resources

- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Prefix Tuning Paper](https://arxiv.org/abs/2101.00190)
- [Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)

## ⚡ Performance Tips

- Start with LoRA for best balance of performance and efficiency
- Use IA³ when memory is extremely constrained
- Combine with 8-bit/4-bit quantization for even more savings
- Experiment with rank values in LoRA (typically 4-64)
- Use gradient checkpointing for very large models

## 📝 License

Please check individual notebook headers for specific licensing information.

## 🌟 Acknowledgments

Built using Hugging Face Transformers and PEFT libraries. Thanks to the open-source community for these incredible tools that democratize access to large language models.

---

**Star ⭐ this repository if you find it useful!**

For questions or discussions, feel free to open an issue.