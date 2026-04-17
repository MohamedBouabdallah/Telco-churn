import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


DEFAULT_API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)
REQUEST_TIMEOUT_SECONDS = 75

YES_NO_OPTIONS = ["No", "Yes"]
INTERNET_SERVICE_OPTIONS = ["DSL", "Fiber optic", "No"]
DEPENDENT_INTERNET_FIELDS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
FIELD_LABELS = {
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
}
DRIVER_LABELS = {
    "Contract_Month-to-month": "Month-to-month contract",
    "OnlineSecurity_No": "No online security",
    "TechSupport_No": "No tech support",
    "MonthlyCharges": "High monthly charges",
    "InternetService_Fiber optic": "Fiber optic internet service",
    "PaperlessBilling_No": "No paperless billing",
    "OnlineBackup_No": "No online backup",
}
PRIORITY_LABELS = {
    "High risk": "High",
    "Moderate risk": "Medium",
    "Low risk": "Low",
}

EXAMPLE_PROFILES = {
    "New fiber customer": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 4,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 96.5,
        "TotalCharges": 386.0,
    },
    "Mid-tenure customer": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 18,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 58.75,
        "TotalCharges": 1057.5,
    },
    "Long-term customer": {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 62,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 63.4,
        "TotalCharges": 3930.8,
    },
}


def call_api(api_base_url, method, path, payload=None):
    """Call the FastAPI backend and return a display-friendly result"""
    url = f"{api_base_url.rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8")
            return {
                "ok": 200 <= response.status < 300,
                "status_code": response.status,
                "data": json.loads(response_body) if response_body else {},
                "error": None,
            }
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8")
        try:
            error_data = json.loads(response_body) if response_body else {}
        except json.JSONDecodeError:
            error_data = {"detail": response_body}

        return {
            "ok": False,
            "status_code": exc.code,
            "data": error_data,
            "error": f"API returned HTTP {exc.code}.",
        }
    except URLError as exc:
        return {
            "ok": False,
            "status_code": None,
            "data": {},
            "error": f"Could not reach the API: {exc.reason}",
        }
    except TimeoutError:
        return {
            "ok": False,
            "status_code": None,
            "data": {},
            "error": (
                "The API request timed out. If the backend is hosted on a free Render "
                "instance, it may be waking up from inactivity. Please try again in a moment."
            ),
        }


def option_index(options, value):
    """Return the index of a value in a list of options."""
    return options.index(value) if value in options else 0


def render_health_status(api_base_url):
    """Display the backend health status in the dashboard"""
    st.subheader("API status")
    health_response = call_api(api_base_url, "GET", "/health")

    if not health_response["ok"]:
        st.error(health_response["error"] or "The API health check failed.")
        st.caption("Start the FastAPI backend before scoring a customer.")
        return health_response

    health = health_response["data"]
    model_loaded = health.get("model_loaded", False)
    if model_loaded:
        st.success("API is online and the churn model is loaded.")
    else:
        st.warning("API is online, but the model is not loaded.")

    status_col, model_col = st.columns(2)
    status_col.metric("Service status", health.get("status", "unknown"))
    model_col.metric("Model loaded", "Yes" if model_loaded else "No")
    return health_response


def build_customer_form(selected_profile):
    """Render the customer form and build the prediction payload"""
    st.subheader("Customer profile")

    with st.form("customer_profile_form"):
        st.markdown("#### Household and customer details")
        profile_col_1, profile_col_2 = st.columns(2)
        gender = profile_col_1.selectbox(
            "Gender",
            ["Female", "Male"],
            index=option_index(["Female", "Male"], selected_profile["gender"]),
        )
        senior_citizen_label = profile_col_2.selectbox(
            "Senior citizen",
            ["No", "Yes"],
            index=selected_profile["SeniorCitizen"],
        )
        partner = profile_col_1.selectbox(
            "Lives with a partner",
            YES_NO_OPTIONS,
            index=option_index(YES_NO_OPTIONS, selected_profile["Partner"]),
        )
        dependents = profile_col_2.selectbox(
            "Has dependents in the household",
            YES_NO_OPTIONS,
            index=option_index(YES_NO_OPTIONS, selected_profile["Dependents"]),
        )
        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=int(selected_profile["tenure"]),
        )

        st.markdown("#### Phone and subscription services")
        phone_service = st.selectbox(
            "Uses phone service",
            YES_NO_OPTIONS,
            index=option_index(YES_NO_OPTIONS, selected_profile["PhoneService"]),
        )
        if phone_service == "Yes":
            multiple_lines = st.selectbox(
                "Has multiple phone lines",
                YES_NO_OPTIONS,
                index=option_index(YES_NO_OPTIONS, selected_profile["MultipleLines"]),
            )
        else:
            multiple_lines = "No phone service"
            st.info("Multiple lines is set to 'No phone service'.")

        st.markdown("#### Internet and support services")
        internet_service = st.selectbox(
            "Internet service",
            INTERNET_SERVICE_OPTIONS,
            index=option_index(
                INTERNET_SERVICE_OPTIONS,
                selected_profile["InternetService"],
            ),
        )
        internet_values = {}
        if internet_service == "No":
            for field in DEPENDENT_INTERNET_FIELDS:
                internet_values[field] = "No internet service"
            st.info("Internet add-ons are set to 'No internet service'.")
        else:
            internet_col_1, internet_col_2 = st.columns(2)
            for index, field in enumerate(DEPENDENT_INTERNET_FIELDS):
                column = internet_col_1 if index % 2 == 0 else internet_col_2
                internet_values[field] = column.selectbox(
                    FIELD_LABELS[field],
                    YES_NO_OPTIONS,
                    index=option_index(YES_NO_OPTIONS, selected_profile[field]),
                )

        st.markdown("#### Contract and billing")
        billing_col_1, billing_col_2 = st.columns(2)
        contract = billing_col_1.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
            index=option_index(
                ["Month-to-month", "One year", "Two year"],
                selected_profile["Contract"],
            ),
        )
        paperless_billing = billing_col_2.selectbox(
            "Uses paperless billing",
            YES_NO_OPTIONS,
            index=option_index(YES_NO_OPTIONS, selected_profile["PaperlessBilling"]),
        )
        payment_method = billing_col_1.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            index=option_index(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                selected_profile["PaymentMethod"],
            ),
        )
        monthly_charges = billing_col_2.number_input(
            "Monthly charges",
            min_value=0.0,
            max_value=200.0,
            value=float(selected_profile["MonthlyCharges"]),
            step=1.0,
        )
        total_charges = st.number_input(
            "Total charges",
            min_value=0.0,
            max_value=10000.0,
            value=float(selected_profile["TotalCharges"]),
            step=10.0,
        )

        submitted = st.form_submit_button("Assess churn risk")

    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen_label == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        **internet_values,
    }

    return submitted, payload


def render_prediction_result(response):
    """Display churn risk results and recommended retention actions"""
    if not response["ok"]:
        st.error("The risk assessment could not be completed. Please try again.")
        return

    data = response["data"]
    probability = data.get("churn_probability")
    risk_segment = data.get("risk_segment", "Unknown")

    st.subheader("Risk assessment")
    probability_col, segment_col = st.columns(2)
    if isinstance(probability, (float, int)):
        probability_col.metric("Estimated churn probability", f"{probability:.1%}")
    else:
        probability_col.metric("Estimated churn probability", "Unavailable")

    segment_col.metric("Retention priority", PRIORITY_LABELS.get(risk_segment, risk_segment))

    if risk_segment == "High risk":
        st.error("Prioritize this customer for retention outreach.")
    elif risk_segment == "Moderate risk":
        st.warning("Monitor this customer and consider a targeted intervention.")
    elif risk_segment == "Low risk":
        st.success("Keep standard engagement and service quality high.")

    st.markdown("#### Main risk factors")
    top_drivers = data.get("top_churn_drivers", [])
    if top_drivers:
        for driver in top_drivers:
            display_driver = DRIVER_LABELS.get(driver, driver)
            st.write(f"- {display_driver}")
    else:
        st.caption("No detailed risk factors are available for this customer.")

    st.markdown("#### Recommended actions")
    recommendations = data.get("recommendation", [])
    if recommendations:
        for recommendation in recommendations:
            st.success(recommendation)
    else:
        st.caption("No retention action suggestion is available for this customer.")


def main():
    """Run the Streamlit dashboard application"""
    st.set_page_config(
        page_title="Telco Churn Dashboard",
        layout="centered",
    )

    st.title("Customer retention decision dashboard")
    st.write(
        "Review a customer profile, assess churn risk and identify the most relevant "
        "retention actions."
    )

    with st.sidebar:
        st.header("Options")
        show_debug = st.checkbox("Show technical details", value=False)

        if show_debug:
            st.markdown("#### Backend connection")
            api_base_url = st.text_input("FastAPI base URL", value=DEFAULT_API_BASE_URL)
            st.caption("Start FastAPI locally before scoring a customer.")
        else:
            api_base_url = DEFAULT_API_BASE_URL

    if show_debug:
        render_health_status(api_base_url)

    st.subheader("Starter profile")
    selected_example = st.selectbox(
        "Choose a customer scenario",
        list(EXAMPLE_PROFILES),
        index=0,
    )
    selected_profile = EXAMPLE_PROFILES[selected_example]

    submitted, payload = build_customer_form(selected_profile)

    if show_debug:
        with st.expander("Raw request payload"):
            st.json(payload)

    if submitted:
        prediction_response = call_api(api_base_url, "POST", "/predict", payload)
        render_prediction_result(prediction_response)

        if show_debug:
            with st.expander("Raw API response", expanded=not prediction_response["ok"]):
                st.json(
                    {
                        "status_code": prediction_response["status_code"],
                        "response": prediction_response["data"],
                    }
                )


if __name__ == "__main__":
    main()
