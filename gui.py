import tkinter as tk
from tkinter import ttk, scrolledtext
from main import response
import threading

class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot")
        self.root.geometry("600x700")
        self.root.configure(bg="#f0f0f0")

        # Frame principal
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Área de chat
        self.chat_area = scrolledtext.ScrolledText(
            self.main_frame,
            wrap=tk.WORD,
            width=50,
            height=30,
            font=("Arial", 10),
            bg="white"
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.chat_area.config(state=tk.DISABLED)

        # Frame para entrada de texto
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.pack(fill=tk.X)

        # Campo de entrada
        self.input_field = ttk.Entry(
            self.input_frame,
            font=("Arial", 10)
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_field.bind("<Return>", self.send_message)

        # Botón de enviar
        self.send_button = ttk.Button(
            self.input_frame,
            text="Enviar",
            command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        # Mensaje de bienvenida
        self.add_message("Bot", "Bienvenido a tu chatbot favorito")

    def add_message(self, sender, message):
        self.chat_area.config(state=tk.NORMAL)
        if sender == "Bot":
            self.chat_area.insert(tk.END, f"{sender}: {message}\n", "bot")
        else:
            self.chat_area.insert(tk.END, f"{sender}: {message}\n", "user")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def send_message(self, event=None):
        message = self.input_field.get().strip()
        if message:
            self.add_message("Tú", message)
            self.input_field.delete(0, tk.END)
            
            # Procesar respuesta en un hilo separado
            threading.Thread(target=self.process_response, args=(message,), daemon=True).start()

    def process_response(self, message):
        # Obtener respuesta del chatbot
        response_text, should_continue = response(message)
        self.add_message("Bot", response_text)
        if not should_continue:
            self.root.after(1000, self.root.destroy)  # Cerrar la ventana después de 1 segundo

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop() 