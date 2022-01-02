from models.qa.mag import MAG
from models.qa.rag import RAG
from models.qa.answer_retriever import AnswerRetriever
from models.qa.answer_generator import AnswerGenerator
from models.qa.question_generator import QuestionGenerator
from models.qa.context_retriever import ContextRetriever
from models.qa.text_encoder_decoder import TextEncoderDecoder
from models.qa.answer_generator_split import AnswerGeneratorSplit

_models = {
    'MAG'           : MAG,
    'RAG'           : RAG,
    'QARetriever'   : AnswerRetriever,
    'AnswerGenerator'   : AnswerGenerator,
    'QuestionGenerator' : QuestionGenerator,
    'ContextRetriever'  : ContextRetriever,
    'TextEncoderDecoder'    : TextEncoderDecoder,
    'AnswerGeneratorSplit'  : AnswerGeneratorSplit
}

