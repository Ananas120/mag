from models.qa.base_qa_generator import BaseQAGenerator

class QuestionGenerator(BaseQAGenerator):
    def __init__(self, * args, input_format = ['{answer}', '{context}'], output_format = '{question}', ** kwargs):
        super().__init__(* args, input_format = input_format, output_format = output_format, ** kwargs)
    
