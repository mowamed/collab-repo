# collab-repo

# Florence-2-large sample usage

This notebook demonstrates how to use the Florence-2-large model for various vision-language tasks.

## Installation

Install the necessary packages:

```bash
!pip install {' '.join(missing_packages)}
!pip install flash_attn einops timm
```

## Usage

1. **Load the model and processor:**

```python
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = 'microsoft/Florence-2-large' model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
```

2. **Define the prediction function:**

```python
def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].cuda(),
      pixel_values=inputs["pixel_values"].cuda(),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer
```

3. **Load an image:**

```python
from PIL import Image import requests

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)
```

4. **Run pre-defined tasks:**

```python

#Captioning
task_prompt = '<CAPTION>'
run_example(task_prompt)

#Object detection
task_prompt = '<OD>'
results = run_example(task_prompt)

#... (plot the results)
#Dense region captioning
task_prompt = '<DENSE_REGION_CAPTION>'
results = run_example(task_prompt)

#... (plot the results)
#... (other tasks)

```

5. **Run tasks with additional inputs:**

```python

#Phrase grounding
task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
results = run_example(task_prompt, text_input="A green car parked in front of a yellow building")

#... (plot the results)
#Referring expression segmentation
task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
results = run_example(task_prompt, text_input="a green car")

#... (plot the results)
#... (other tasks)

```

6. **OCR tasks:**

```python
#Load an image with text
url = "https://cursivehandwriting.co.in/wp-content/uploads/2023/11/f9b3033e7a38c727af188a4604e0c0c4-e1700722488449.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

#OCR
task_prompt = '<OCR>'
run_example(task_prompt)
print(results)
# ocr results format
# {'<OCR>': '\nHandwriting is an essential skill for both childrenand adults (Fider & Majnemer, 2007). Even inthe age of technology, it remains the primary toolof communication and knowledge assessment forstudents in the classroom. The demands for it are great,whether in the classroom or beyond. A 1992 studyMcHale & Cernak) found that 85 percent of all finemotor time in second-fourth-and sixth-grade classroomwas spent on paper and pencil activities. A morerecent study (Mari, Cernack, John & Henderson, 2003)noted that kindergarten children are now spending42 percent of their fine motor time on paper anpencil activities.\n'}

# OCR with region
task_prompt = ''
results = run_example(task_prompt)

#... (plot the results)

```

## Results

Execute the code yourself to see the output and generated images for each task.
