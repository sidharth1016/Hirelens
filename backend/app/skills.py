"""Skill taxonomy + extraction.

Each canonical skill maps to a category and a list of aliases. Extraction is
regex-based with word boundaries, so "React.js", "ReactJS" and "react" all map
to the canonical "React".
"""
from __future__ import annotations

import re
from functools import lru_cache

# canonical -> (category, [aliases])
SKILLS: dict[str, tuple[str, list[str]]] = {
    # Programming languages
    "Python": ("Languages", ["python", "python3"]),
    "Java": ("Languages", ["java"]),
    "C++": ("Languages", ["c++", "cpp"]),
    "C": ("Languages", [r"\bc\b(?!\+|#)"]),
    "C#": ("Languages", ["c#", "csharp"]),
    "JavaScript": ("Languages", ["javascript", "es6"]),
    "TypeScript": ("Languages", ["typescript"]),
    "Go": ("Languages", ["golang"]),
    "Rust": ("Languages", ["rust"]),
    "Kotlin": ("Languages", ["kotlin"]),
    "Swift": ("Languages", ["swift"]),
    "R": ("Languages", [r"\br\b(?= programming| language|,|/| and)"]),
    "Scala": ("Languages", ["scala"]),
    "SQL": ("Languages", ["sql"]),
    "Bash": ("Languages", ["bash", "shell scripting"]),
    "MATLAB": ("Languages", ["matlab"]),
    "PHP": ("Languages", ["php"]),
    "Ruby": ("Languages", ["ruby"]),
    # Core ML / DS
    "Machine Learning": ("AI/ML", ["machine learning", "ml", "ai/ml", "aiml"]),
    "Deep Learning": ("AI/ML", ["deep learning", "dl"]),
    "Artificial Intelligence": ("AI/ML", ["artificial intelligence", "ai"]),
    "NLP": ("AI/ML", ["nlp", "natural language processing"]),
    "Computer Vision": ("AI/ML", ["computer vision"]),
    "Reinforcement Learning": ("AI/ML", ["reinforcement learning"]),
    "Statistics": ("AI/ML", ["statistics", "statistical analysis"]),
    "Data Analysis": ("AI/ML", ["data analysis", "data analytics"]),
    "Data Science": ("AI/ML", ["data science"]),
    "Feature Engineering": ("AI/ML", ["feature engineering"]),
    "Model Deployment": ("AI/ML", ["model deployment", "deploying models", "model serving"]),
    "Time Series": ("AI/ML", ["time series", "forecasting"]),
    "Recommendation Systems": ("AI/ML", ["recommendation system", "recommender system", "recommendation systems"]),
    "Classification": ("AI/ML", ["classification"]),
    "Regression": ("AI/ML", ["regression"]),
    "Clustering": ("AI/ML", ["clustering", "k-means", "kmeans"]),
    "Neural Networks": ("AI/ML", ["neural network", "neural networks"]),
    "CNN": ("AI/ML", ["cnn", "convolutional neural network"]),
    "RNN": ("AI/ML", ["rnn", "lstm", "gru"]),
    "Transformers": ("AI/ML", ["transformers", "transformer models", "attention mechanism"]),
    "Object Detection": ("AI/ML", ["object detection", "yolo"]),
    "Image Processing": ("AI/ML", ["image processing"]),
    "Speech Recognition": ("AI/ML", ["speech recognition", "speech-to-text", "asr"]),
    "MLOps": ("AI/ML", ["mlops"]),
    "A/B Testing": ("AI/ML", ["a/b testing", "ab testing"]),
    # ML libraries
    "NumPy": ("ML Libraries", ["numpy"]),
    "Pandas": ("ML Libraries", ["pandas"]),
    "Scikit-learn": ("ML Libraries", ["scikit-learn", "sklearn", "scikit learn"]),
    "TensorFlow": ("ML Libraries", ["tensorflow", "tf2"]),
    "Keras": ("ML Libraries", ["keras"]),
    "PyTorch": ("ML Libraries", ["pytorch", "torch"]),
    "OpenCV": ("ML Libraries", ["opencv", "cv2"]),
    "XGBoost": ("ML Libraries", ["xgboost"]),
    "LightGBM": ("ML Libraries", ["lightgbm"]),
    "Matplotlib": ("ML Libraries", ["matplotlib"]),
    "Seaborn": ("ML Libraries", ["seaborn"]),
    "Plotly": ("ML Libraries", ["plotly"]),
    "SciPy": ("ML Libraries", ["scipy"]),
    "NLTK": ("ML Libraries", ["nltk"]),
    "spaCy": ("ML Libraries", ["spacy"]),
    "Hugging Face": ("ML Libraries", ["hugging face", "huggingface", "hf transformers"]),
    "Jupyter": ("ML Libraries", ["jupyter", "jupyter notebook"]),
    "MLflow": ("ML Libraries", ["mlflow"]),
    "ONNX": ("ML Libraries", ["onnx"]),
    "Streamlit": ("ML Libraries", ["streamlit"]),
    "Gradio": ("ML Libraries", ["gradio"]),
    # Generative AI
    "Generative AI": ("GenAI", ["generative ai", "genai", "gen ai"]),
    "LLMs": ("GenAI", ["llm", "llms", "large language model", "large language models"]),
    "LangChain": ("GenAI", ["langchain"]),
    "LlamaIndex": ("GenAI", ["llamaindex", "llama index"]),
    "LangGraph": ("GenAI", ["langgraph"]),
    "RAG": ("GenAI", ["rag", "retrieval augmented generation", "retrieval-augmented generation"]),
    "Prompt Engineering": ("GenAI", ["prompt engineering", "prompting"]),
    "OpenAI API": ("GenAI", ["openai", "gpt-4", "gpt-4o", "chatgpt api", "gpt"]),
    "Gemini API": ("GenAI", ["gemini"]),
    "Claude API": ("GenAI", ["claude", "anthropic"]),
    "Fine-tuning": ("GenAI", ["fine-tuning", "fine tuning", "finetuning", "lora", "peft"]),
    "Embeddings": ("GenAI", ["embeddings", "embedding models", "sentence transformers"]),
    "Vector Databases": ("GenAI", ["vector database", "vector databases", "vector db", "vector store"]),
    "Pinecone": ("GenAI", ["pinecone"]),
    "FAISS": ("GenAI", ["faiss"]),
    "ChromaDB": ("GenAI", ["chromadb", "chroma"]),
    "AI Agents": ("GenAI", ["ai agent", "ai agents", "agentic", "autonomous agents"]),
    "Chatbots": ("GenAI", ["chatbot", "chatbots", "conversational ai"]),
    "Stable Diffusion": ("GenAI", ["stable diffusion", "diffusion models"]),
    "Ollama": ("GenAI", ["ollama"]),
    # Backend / web
    "FastAPI": ("Backend & Web", ["fastapi"]),
    "Flask": ("Backend & Web", ["flask"]),
    "Django": ("Backend & Web", ["django"]),
    "Node.js": ("Backend & Web", ["node.js", "nodejs"]),
    "Express": ("Backend & Web", ["express", "express.js"]),
    "React": ("Backend & Web", ["react", "react.js", "reactjs"]),
    "Next.js": ("Backend & Web", ["next.js", "nextjs"]),
    "Vue": ("Backend & Web", ["vue", "vue.js"]),
    "Angular": ("Backend & Web", ["angular"]),
    "HTML": ("Backend & Web", ["html", "html5"]),
    "CSS": ("Backend & Web", ["css", "css3"]),
    "Tailwind CSS": ("Backend & Web", ["tailwind", "tailwindcss"]),
    "REST APIs": ("Backend & Web", ["rest api", "rest apis", "restful"]),
    "GraphQL": ("Backend & Web", ["graphql"]),
    "WebSockets": ("Backend & Web", ["websocket", "websockets"]),
    "Microservices": ("Backend & Web", ["microservices"]),
    # Data / DBs
    "MySQL": ("Databases", ["mysql"]),
    "PostgreSQL": ("Databases", ["postgresql", "postgres"]),
    "MongoDB": ("Databases", ["mongodb", "mongo"]),
    "Redis": ("Databases", ["redis"]),
    "SQLite": ("Databases", ["sqlite"]),
    "Firebase": ("Databases", ["firebase"]),
    "Supabase": ("Databases", ["supabase"]),
    "Elasticsearch": ("Databases", ["elasticsearch"]),
    "Apache Spark": ("Databases", ["spark", "pyspark", "apache spark"]),
    "Hadoop": ("Databases", ["hadoop"]),
    "Kafka": ("Databases", ["kafka"]),
    "Airflow": ("Databases", ["airflow"]),
    "ETL": ("Databases", ["etl", "data pipelines", "data pipeline"]),
    "Snowflake": ("Databases", ["snowflake"]),
    "Power BI": ("Databases", ["power bi", "powerbi"]),
    "Tableau": ("Databases", ["tableau"]),
    "Excel": ("Databases", ["excel"]),
    # Cloud / DevOps
    "Docker": ("Cloud & DevOps", ["docker", "containerization", "containerised", "containerized"]),
    "Kubernetes": ("Cloud & DevOps", ["kubernetes", "k8s"]),
    "AWS": ("Cloud & DevOps", ["aws", "amazon web services", "sagemaker", "ec2", "s3"]),
    "GCP": ("Cloud & DevOps", ["gcp", "google cloud", "vertex ai"]),
    "Azure": ("Cloud & DevOps", ["azure"]),
    "Git": ("Cloud & DevOps", ["git", "github", "gitlab"]),
    "CI/CD": ("Cloud & DevOps", ["ci/cd", "github actions", "jenkins"]),
    "Linux": ("Cloud & DevOps", ["linux", "unix"]),
    "Vercel": ("Cloud & DevOps", ["vercel"]),
    "Nginx": ("Cloud & DevOps", ["nginx"]),
    "Terraform": ("Cloud & DevOps", ["terraform"]),
    # Automation / product
    "Workflow Automation": ("Automation", ["workflow automation", "automation", "automations", "n8n", "zapier", "make.com"]),
    "WhatsApp API": ("Automation", ["whatsapp api", "whatsapp"]),
    "CRM": ("Automation", ["crm", "hubspot", "salesforce"]),
    "Web Scraping": ("Automation", ["web scraping", "beautifulsoup", "selenium", "scrapy"]),
    "Selenium": ("Automation", ["selenium"]),
    "Playwright": ("Automation", ["playwright"]),
    # CS fundamentals
    "Data Structures": ("CS Fundamentals", ["data structures", "dsa"]),
    "Algorithms": ("CS Fundamentals", ["algorithms"]),
    "OOP": ("CS Fundamentals", ["oop", "object oriented programming", "object-oriented"]),
    "DBMS": ("CS Fundamentals", ["dbms"]),
    "Operating Systems": ("CS Fundamentals", ["operating systems"]),
    "Computer Networks": ("CS Fundamentals", ["computer networks", "networking"]),
    "System Design": ("CS Fundamentals", ["system design"]),
    "Unit Testing": ("CS Fundamentals", ["unit testing", "pytest", "unittest"]),
    # Soft skills (the Exora JD asks for these explicitly)
    "Communication": ("Soft Skills", ["communication", "communication skills"]),
    "Teamwork": ("Soft Skills", ["teamwork", "team player", "collaboration"]),
    "Problem Solving": ("Soft Skills", ["problem solving", "problem-solving"]),
    "Leadership": ("Soft Skills", ["leadership", "co-founder", "founder", "led a team"]),
    "Startup Experience": ("Soft Skills", ["startup", "start-up", "fast-paced"]),
    "Client Management": ("Soft Skills", ["client management", "client communication", "stakeholder"]),
    "Pydantic": ("Backend & Web", ["pydantic"]),
    "Celery": ("Backend & Web", ["celery"]),
    "Postman": ("Cloud & DevOps", ["postman"]),
    "Figma": ("Backend & Web", ["figma"]),
    "Agile": ("Soft Skills", ["agile", "scrum"]),
    "Video Editing": ("Soft Skills", ["video editing", "premiere pro", "after effects"]),
}


def _alias_to_pattern(alias: str) -> str:
    if alias.startswith(r"\b") or "(?" in alias:
        return alias  # already a regex
    esc = re.escape(alias)
    # custom boundaries so "c++" / "node.js" / "c#" work
    return rf"(?<![a-z0-9+#]){esc}(?![a-z0-9+#])"


@lru_cache(maxsize=1)
def _compiled() -> list[tuple[str, re.Pattern]]:
    out = []
    for canon, (_, aliases) in SKILLS.items():
        pat = "|".join(_alias_to_pattern(a) for a in aliases)
        out.append((canon, re.compile(pat, re.IGNORECASE)))
    return out


def extract_skills(text: str) -> set[str]:
    """Return the set of canonical skills mentioned in `text`."""
    text = text or ""
    found = {canon for canon, pat in _compiled() if pat.search(text)}
    # Implications: knowing a specific tool implies the umbrella skill.
    implies = {
        "Machine Learning": {"Scikit-learn", "XGBoost", "Classification", "Regression", "Deep Learning"},
        "Generative AI": {"LangChain", "RAG", "LLMs", "OpenAI API", "Gemini API", "Claude API", "Prompt Engineering"},
        "LLMs": {"OpenAI API", "Gemini API", "Claude API", "LangChain"},
        "NLP": {"NLTK", "spaCy", "Transformers"},
        "Computer Vision": {"OpenCV", "CNN", "Object Detection"},
    }
    for umbrella, children in implies.items():
        if found & children:
            found.add(umbrella)
    return found


def category_of(skill: str) -> str:
    return SKILLS.get(skill, ("Other", []))[0]
