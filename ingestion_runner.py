from ingestion.news_fetcher import fetch_news, save_news
from preprocessing.preprocess import process_news
from embeddings.embed_store import store_embeddings

print("\nFETCHING NEWS")
news = fetch_news()
save_news(news)

print("\nPROCESSING")
process_news()

print("\nBUILDING VDB")
store_embeddings()

print("\n✅ READY FOR DEMO")