import streamlit as st
import requests
import json

# Title
st.title("IBM API Key Token Generator")

# Input for API key
api_key = st.text_input("Enter your IBM API Key", type="password")

# Submit button
if st.button("Get Token"):
    if not api_key:
        st.warning("Please enter an API key.")
    else:
        # API URL and headers
        url = "https://iam.platform.saas.ibm.com/siusermgr/api/1.0/apikeys/token"
        headers = {
            "accept": "application/json",
            "content-type": "application/json"
        }
        data = {
            "apikey": api_key
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                token = response.json().get("token")
                if token:
                    st.success("Token retrieved successfully!")
                    st.code(token, language="text")
                else:
                    st.error("Token not found in response.")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
