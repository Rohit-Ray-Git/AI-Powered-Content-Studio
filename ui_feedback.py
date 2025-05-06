import streamlit as st

def show_error_message(message, debug_info=None):
    st.error(message)
    if debug_info:
        with st.expander("Show debug info"):
            st.write(debug_info)

def show_success_message(message):
    st.success(message)

def show_warning_message(message):
    st.warning(message)

def show_info_message(message):
    st.info(message) 