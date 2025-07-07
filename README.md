
# How to run locally

Create a virtual envrionment (optional)
![Static Badge](https://img.shields.io/badge/-recommended-darkgreen.svg)

```bash
  python -m venv venv
```

Activate venv

```bash
  .\venv\Scripts\activate 
```

Install dependencies 

```bash
  pip install -r requirements.txt
```

Start the server with hot reload

```bash
  uvicorn main:app --reload 
```


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`OPENAI_API_KEY`

`PINECONE_API_KEY`

`PINECONE_ENVIRONMENT`

`PINECONE_INDEX_NAME`

`FIRECRAWL_API_KEY`

## Tech Stack

**Server:** Python, FastAPI, Firecrawl

