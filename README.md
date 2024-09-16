# Machine Translation using MTet

We use [MTet](https://huggingface.co/VietAI/envit5-translation), which stands for **Multi-domain Translation for English and Vietnamese**, based on the `VietAI/envit5-translation` model.

## Installation

To use this translation model, you need to install the `transformers` library from Hugging Face. You can install it using pip:

```bash
pip install transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "VietAI/envit5-translation"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.cuda()

# Define input texts
inputs = [
    "vi: VietAI là tổ chức phi lợi nhuận với sứ mệnh ươm mầm tài năng về trí tuệ nhân tạo và xây dựng một cộng đồng các chuyên gia trong lĩnh vực trí tuệ nhân tạo đẳng cấp quốc tế tại Việt Nam.",
    "vi: Theo báo cáo mới nhất của Linkedin về danh sách việc làm triển vọng với mức lương hấp dẫn năm 2020, các chức danh công việc liên quan đến AI như Chuyên gia AI (Artificial Intelligence Specialist), Kỹ sư ML (Machine Learning Engineer) đều xếp thứ hạng cao.",
    "en: Our teams aspire to make discoveries that impact everyone, and core to our approach is sharing our research and tools to fuel progress in the field.",
    "en: We're on a journey to advance and democratize artificial intelligence through open source and open science."
]

# Generate translation
outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cuda'), max_length=512)

# Print the translation results
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
