"""API client for backend communication."""
import os
import requests
from typing import Dict, Any, Optional
import streamlit as st


class APIClient:
    """Client for communicating with the backend API."""

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize API client.

        Args:
            base_url: Base URL for the backend API
        """
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://fastapi:8000")
        # Ensure no trailing slash
        self.base_url = self.base_url.rstrip("/")

    def predict_new_client(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get prediction for a new client.

        Args:
            client_data: Dictionary with client features

        Returns:
            Prediction response from API

        Raises:
            requests.HTTPError: If API request fails
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/predict/new",
                json=client_data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            st.error("⚠️ Request timed out. Please try again.")
            raise
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to backend. Please check if the service is running.")
            raise
        except requests.exceptions.HTTPError as e:
            st.error(f"⚠️ API Error: {e.response.text}")
            raise

    def predict_existing_client(self, client_id: int) -> Dict[str, Any]:
        """
        Get prediction for an existing client.

        Args:
            client_id: Client ID (row index)

        Returns:
            Prediction response from API

        Raises:
            requests.HTTPError: If API request fails
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/predict/existing",
                json={"id_client": client_id},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            st.error("⚠️ Request timed out. Please try again.")
            raise
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to backend. Please check if the service is running.")
            raise
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                st.error(f"⚠️ Client ID {client_id} not found.")
            else:
                st.error(f"⚠️ API Error: {e.response.text}")
            raise

    def get_health(self) -> Dict[str, Any]:
        """
        Check backend health status.

        Returns:
            Health status response

        Raises:
            requests.HTTPError: If health check fails
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return {"status": "unhealthy", "error": "Cannot reach backend"}

    def get_client_count(self) -> Dict[str, Any]:
        """
        Get the number of clients in the dataset.

        Returns:
            Client count information

        Raises:
            requests.HTTPError: If request fails
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/clients/count",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.warning(f"⚠️ Could not retrieve client count: {e}")
            return {"count": "unknown"}


# Global API client instance
@st.cache_resource
def get_api_client() -> APIClient:
    """Get cached API client instance."""
    return APIClient()
