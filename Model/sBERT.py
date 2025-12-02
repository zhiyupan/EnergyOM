from sentence_transformers import SentenceTransformer
import tensorflow as tf


class SentenceTransformers(object):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        '''
        paraphrase-MiniLM-L6-v2
        '''
        self.model = SentenceTransformer(model_name)

    def embedding(self, sentences):
        embeddings = self.model.encode(sentences)
        tensor = tf.convert_to_tensor(embeddings)
        return tensor
