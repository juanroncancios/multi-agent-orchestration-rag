# Sistema Multi-Agente con RAG y Trazabilidad en Langfuse
Este proyecto implementa una arquitectura de Orquestación Multi-Agente impulsada por LangChain y OpenAI, integrada con un sistema de Recuperación Aumentada por Generación (RAG) sobre bases vectoriales independientes en FAISS y monitoreada en tiempo real mediante Langfuse.

🏗️ Arquitectura del Sistema
El sistema utiliza un Orquestador Principal (Router Agent) encargado de clasificar la intención de la consulta del usuario y redirigirla al agente/retriever especializado correspondiente:

Agente HR: Consulta políticas de recursos humanos (vacaciones, licencias, beneficios).

Agente TECH: Resuelve problemas de soporte técnico (VPN, contraseñas, políticas de seguridad).

Agente FINANCE: Procesa consultas de gastos, reembolsos, facturación y límites diarios.

Plaintext
                  ┌──────────────────────┐
                  │ Consulta del Usuario │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │ Router / Orquestador │
                  └──────────┬───────────┘
        ┌────────────────────┼────────────────────┐
        │ [HR]               │ [TECH]             │ [FINANCE]
┌───────▼───────┐    ┌───────▼───────┐    ┌───────▼───────┐
│ Vectorstore HR│    │Vectorstore TECH│   │Vectorstore FIN│
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        └────────────────────┼────────────────────┘
                             │
                   ┌─────────▼────────┐
                   │ Respuesta Final  │
                   └──────────────────┘
🛠️ Requisitos Previos
Python 3.10+

Cuenta en OpenAI (API Key)

Cuenta en Langfuse (Public Key, Secret Key y Host)

⚙️ Configuración e Instalación
Clonar el repositorio:

Bash
git clone https://github.com/juanroncancios/multi-agent-orchestration-rag.git
cd proyect
Crear y activar entorno virtual:

PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
Instalar dependencias:

Bash
pip install -r requirements.txt
Configurar variables de entorno:
Crea un archivo .env en la raíz del proyecto tomando como base .env.example:

Fragmento de código
OPENAI_API_KEY=tu_openai_api_key
LANGFUSE_PUBLIC_KEY=tu_langfuse_public_key
LANGFUSE_SECRET_KEY=tu_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
🚀 Ejecución del Proyecto
Para correr la suite de pruebas multi-agente:

PowerShell
python multi-agent_orchestration.py
El script ejecutará automáticamente 10 casos de prueba evaluando el enrutamiento correcto y la extracción contextual desde las bases vectoriales.

📊 Observabilidad y Trazabilidad (Langfuse)
Todas las ejecuciones, llamadas a la API de OpenAI, latencias y decisiones del orquestador son enviadas en tiempo real a Langfuse.

Para visualizar los traces:

Accede al Dashboard de Langfuse.

Navega a Traces para auditar el flujo de decisión de cada agente y evaluar el desempeño de la recuperación de contexto (RAG).