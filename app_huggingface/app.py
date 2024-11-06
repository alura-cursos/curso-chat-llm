from litellm import completion
import requests
import json
import gradio as gr
import pandas as pd
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from transformers import pipeline
import numpy as np

load_dotenv()

def info_geologia(query):
    PINECONE_API = os.getenv('PINECONE_API')
    pc = Pinecone(api_key=PINECONE_API)
    index = pc.Index("geologia")
    x = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )
    results = index.query(
        namespace="ns1",
        vector=x[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    return results

def previsao_do_tempo(city, country):
    WEATHER_API = os.getenv('WEATHER_API')
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country}&APPID={WEATHER_API}&lang=pt_br&units=metric"
    response = requests.get(url)
    data = response.json()

    return json.dumps(data)


def verificar_tempestade_solar():
    url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        latest_kp = float(data[-1][1])  # O último valor Kp
        if latest_kp >= 5:
            return f"Alerta de tempestade solar! Índice Kp atual: {latest_kp}"
        else:
            return f"Sem tempestade solar no momento. Índice Kp atual: {latest_kp}"
    else:
        return "Não foi possível obter informações sobre tempestades solares no momento."

def extrair_sismos():
    # Fazer a requisição para obter o conteúdo da página
    url = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.csv'

    df = pd.read_csv(url)

    # Retornar o DataFrame com os dados
    return df


tools = [
        {
            "type": "function",
            "function": {
                "name": "previsao_do_tempo",
                "description": "Retorna a previsão do tempo em uma cidade específica",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Nome da cidade",
                        },
                        "country": {
                            "type": "string",
                            "description": "Sigla do país",
                        },
                    },
                    "required": ["city", "country"],
                },
            }

        },

      {
        "type": "function",
        "function": {
            "name": "verificar_tempestade_solar",
            "description": "Verifica se há uma tempestade solar em andamento",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
      },
      {
        "type": "function",
        "function": {
            "name": "extrair_sismos",
            "description": "Extrai dados de sismos da USGS",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
      }
]


# Função para chamar a API com o histórico de mensagens
def call_groq_api(messages, model="groq/llama3-groq-70b-8192-tool-use-preview"):
    global tools
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    response = completion(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        api_key=GROQ_API_KEY,
    )
    resposta_texto = response.choices[0].message
    chamada_ferramentas = resposta_texto.tool_calls
    if chamada_ferramentas:
      available_functions = {
        "previsao_do_tempo": previsao_do_tempo,
        "verificar_tempestade_solar": verificar_tempestade_solar,
        "extrair_sismos": extrair_sismos
      }
      for tool_call in chamada_ferramentas:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)

        match function_name:
          case "previsao_do_tempo":

            function_response = function_to_call(
                city=function_args.get("city"),
                country=function_args.get("country"),
            )
          case "verificar_tempestade_solar":
            function_response = function_to_call()
          case "extrair_sismos":
            function_response = function_to_call()
        return function_response

    else:
      return resposta_texto.content


def response(message, history):
    messages = [{"role": "system", "content": """
    Você é o Chat da Terra e do Universo e responde em português brasileiro
    perguntas sobre a previsão do tempo na Terra e do espaço próximo à Terra, além de informações sobre terremotos.
    """}]

    # Adicionar o histórico anterior ao histórico de mensagens
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    # Adicionar a nova mensagem do usuário
    messages.append({"role": "user", "content": message})

    # Verificar se o tema é geologia
    if "geologia" in message.lower():
        resposta = info_geologia(message)
        geologia_info = f"Informações adicionais sobre geologia: {resposta.matches[0].metadata['text']}"
        messages.append({"role": "system", "content": geologia_info})

    # Obter a resposta do modelo
    model_response = call_groq_api(messages)

    # Se a resposta for um DataFrame, convertê-la para texto
    if isinstance(model_response, pd.DataFrame):
        texto_corrido = ""
        for index, row in model_response.iterrows():
            texto_corrido += f"Evento {index + 1}: Magnitude {row['mag']}, Local {row['place']}, Tempo {row['time']}\n"
        model_response = texto_corrido

    # Retornar a resposta como string para Gradio
    return model_response

transcritor = pipeline("automatic-speech-recognition",model="openai/whisper-base",generate_kwargs = {"task":"transcribe", "language":"<|pt|>"})
def transcricao(audio):
    sr, y = audio
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcritor({"sampling_rate": sr, "raw": y})["text"]

def responde_audio(audio):
    messages = [{"role": "system", "content": """
    Você é o Chat da Terra e do Universo e responde em português brasileiro
    perguntas sobre a previsão do tempo na Terra e do espaço próximo à Terra, além de informações sobre terremotos.
    """}]
    message = transcricao(audio)
   

    # Adicionar a nova mensagem do usuário
    messages.append({"role": "user", "content": message})

    # Verificar se o tema é geologia
    if "geologia" in message.lower():
        resposta = info_geologia(message)
        geologia_info = f"Informações adicionais sobre geologia: {resposta.matches[0].metadata['text']}"
        messages.append({"role": "system", "content": geologia_info})

    # Obter a resposta do modelo
    model_response = call_groq_api(messages)

    # Se a resposta for um DataFrame, convertê-la para texto
    if isinstance(model_response, pd.DataFrame):
        texto_corrido = ""
        for index, row in model_response.iterrows():
            texto_corrido += f"Evento {index + 1}: Magnitude {row['mag']}, Local {row['place']}, Tempo {row['time']}\n"
        model_response = texto_corrido

    # Retornar a resposta como string para Gradio
    return model_response
    

with gr.Blocks() as demo:
    with gr.Tab("Chat da Terra e do Universo"):
        gr.ChatInterface(
            response,
            title='🌍☀️🌧️ Chat da Terra e do Universo',
            textbox=gr.Textbox(placeholder="Digite sua mensagem aqui..."),
            submit_btn=gr.Button("Enviar")

        )

    with gr.Tab("Assitente de áudio"):
        gr.Interface(
            fn=responde_audio,
            inputs=[gr.Audio(sources="microphone")],
            outputs=["text"],
            title="Assistente de Áudio"
        )

demo.launch(debug=True)