from models.qa.base_qa_generator import BaseQAGenerator

class AnswerGenerator(BaseQAGenerator):
    def __init__(self, * args, input_format = ['{question}', '{context}'], output_format = '{answer}', ** kwargs):
        super().__init__(* args, input_format = input_format, output_format = output_format, ** kwargs)
    
