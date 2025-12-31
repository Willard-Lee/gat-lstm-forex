"""Clear Streamlit cache to force reload of the model."""
import streamlit as st
import shutil
import os

# Clear Streamlit cache directory
cache_dir = os.path.expanduser("~/.streamlit/cache")
if os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir)
        print(f"✅ Cleared Streamlit cache at: {cache_dir}")
    except Exception as e:
        print(f"Error clearing cache: {e}")
else:
    print("No cache directory found")

print("\n🔄 Now restart your Streamlit app:")
print("   streamlit run app.py")
