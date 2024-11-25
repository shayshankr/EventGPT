"""
1.	Visual Question Answering (VQA): These tasks ask the model to answer open-ended questions based on images. Datasets such as VQA-v2 and GQA test the model’s ability to understand and reason about visual content in conjunction with textual queries ￼ ￼.
2.	Text-based Visual Question Answering (TextVQA): In these datasets, the model must read and interpret text that appears within images (e.g., street signs or book covers) and then answer relevant questions. This type is crucial for testing how well the model can handle text extraction and interpretation from visual contexts ￼ ￼.
3.	Image Captioning: The task here is to generate a detailed textual description of an image. Datasets like MS-COCO and Conceptual Captions provide image-text pairs where the model needs to describe the visual content in coherent sentences ￼.
4.	Multiple Choice Questions (MCQ): This involves selecting the correct answer from multiple choices based on visual inputs. Such questions test the model’s ability to reason and select the most appropriate response from a set of possible answers ￼.
5.	Complex Reasoning: These tasks involve deeper inference where the model must not only identify objects but also deduce relationships, actions, or outcomes based on visual and textual cues. The integration of datasets like ScienceQA introduces this level of reasoning ￼.
"""

"""
image_caption_system = '''
### Task:
You are an expert in image description and understanding. You are requested to create a detailed description for the image sent to you.

#### Guidelines For Image Description:
- Analyze the image thoroughly and describe it in detail.
- Include as many details as possible, such as shapes, textures, objects, people, actions, scenes, and backgrounds.
- If text appears in the image, you must transcribe the text in its original language and provide an English translation in parentheses). For example: 书本 (book). Additionally, explain the meaning of the text within its context.
- When referring to people, use their characteristics, such as clothing or appearance, to distinguish different individuals.
- Be confident in your description; avoid using uncertain language like "maybe", "possibly", or "might be".
### **IMPORTANT** 
1. Provide a comprehensive and vivid description that would allow someone to visualize the image without seeing it.
2. Please do not include any information related to color

### Output Format:
Question: [First question about the image]
Answer: [Answer to the first question]

Question: [Second question about the image]
Answer: [Answer to the second question]

Question: [Third question about the image]
Answer: [Answer to the third question]
"""

