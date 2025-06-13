import json
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class Chatbot:
    def __init__(self):
        # Cargar ambos archivos JSON
        with open('conceptos.json', 'r', encoding='utf-8') as f:
            self.conceptos = json.load(f)
        with open('ejemplos.json', 'r', encoding='utf-8') as f:
            self.ejemplos = json.load(f)
        
        # Inicializar el stemmer para español
        self.stemmer = SnowballStemmer('spanish')
        self.stop_words = set(stopwords.words('spanish'))
        
        # Preparar los datos
        self.preparar_datos()

    def preparar_datos(self):
        # Preparar datos de conceptos
        self.conceptos_patterns = []
        self.conceptos_responses = []
        self.conceptos_tags = []
        
        for intent in self.conceptos['intents']:
            for pattern in intent['patterns']:
                self.conceptos_patterns.append(pattern)
                self.conceptos_responses.append(intent['responses'][0])
                self.conceptos_tags.append(intent['tag'])
        
        # Preparar datos de ejemplos
        self.ejemplos_patterns = []
        self.ejemplos_responses = []
        self.ejemplos_tags = []
        
        for intent in self.ejemplos['intents']:
            for pattern in intent['patterns']:
                self.ejemplos_patterns.append(pattern)
                self.ejemplos_responses.append(intent['responses'][0])
                self.ejemplos_tags.append(intent['tag'])

    def preprocesar_texto(self, texto):
        # Tokenizar y convertir a minúsculas
        tokens = word_tokenize(texto.lower())
        # Eliminar stopwords y aplicar stemming
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return tokens

    def calcular_similitud(self, texto1, texto2):
        # Tokenizar y preprocesar ambos textos
        tokens1 = set(self.preprocesar_texto(texto1))
        tokens2 = set(self.preprocesar_texto(texto2))
        
        # Calcular similitud de Jaccard
        if not tokens1 or not tokens2:
            return 0
        return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

    def obtener_respuesta(self, texto_usuario):
        # Buscar en conceptos
        similitudes_conceptos = [self.calcular_similitud(texto_usuario, pattern) 
                               for pattern in self.conceptos_patterns]
        mejor_idx_concepto = np.argmax(similitudes_conceptos)
        mejor_similitud_concepto = similitudes_conceptos[mejor_idx_concepto]
        tag_concepto = self.conceptos_tags[mejor_idx_concepto]
        
        # Buscar en ejemplos
        similitudes_ejemplos = [self.calcular_similitud(texto_usuario, pattern) 
                              for pattern in self.ejemplos_patterns]
        mejor_idx_ejemplo = np.argmax(similitudes_ejemplos)
        mejor_similitud_ejemplo = similitudes_ejemplos[mejor_idx_ejemplo]
        tag_ejemplo = self.ejemplos_tags[mejor_idx_ejemplo]
        
        respuesta = ""
        
        if mejor_similitud_concepto > 0.3:
            respuesta += "📚 **Concepto:**\n"
            respuesta += self.conceptos_responses[mejor_idx_concepto] + "\n\n"
            # Solo mostrar ejemplo si el tag coincide
            if mejor_similitud_ejemplo > 0.3 and tag_concepto == tag_ejemplo:
                respuesta += "💻 **Ejemplo Práctico:**\n"
                respuesta += self.ejemplos_responses[mejor_idx_ejemplo]
        elif mejor_similitud_ejemplo > 0.3:
            respuesta += "💻 **Ejemplo Práctico:**\n"
            respuesta += self.ejemplos_responses[mejor_idx_ejemplo]
        else:
            return "Lo siento, no tengo información específica sobre ese tema. ¿Podrías reformular tu pregunta?"
        
        return respuesta

def main():
    chatbot = Chatbot()
    print("¡Hola! Soy un chatbot especialista en funciones anónimas. ¿En qué puedo ayudarte?")
    
    while True:
        texto_usuario = input("Tú: ")
        if texto_usuario.lower() in ['salir', 'adios', 'chao']:
            print("Chatbot: ¡Hasta luego! Que tengas un excelente día.")
            break
        
        respuesta = chatbot.obtener_respuesta(texto_usuario)
        print("\nChatbot:", respuesta, "\n")

if __name__ == "__main__":
    main() 