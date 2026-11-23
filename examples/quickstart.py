from openai import OpenAI

client = OpenAI(base_url="http://localhost:7860/v1", api_key="your-strong-gateway-secret")
print(client.models.list())
# Configure a real enabled model in gateway/catalog.json before generating.
