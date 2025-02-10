# Use a Python image
FROM python:3.10

# Set working dir inside the container
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose port for Streamlit
EXPOSE 8501

# Define env vars as placeholders, values are injected at runtime
ENV DB_USER=""
ENV DB_PASSWORD=""
ENV DB_HOST=""
ENV DB_NAME=""
ENV API_KEY=""
ENV DB_URI=""

# Run the Streamlit app
CMD ["streamlit", "run", "app/chatbot_with_guardrails.py"]