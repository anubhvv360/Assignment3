import streamlit as st
import pandas as pd
import io
from langchain.llms import GooglePalm
from langchain import PromptTemplate, LLMChain

# Initialize GooglePalm LLM with the API key from Streamlit Secrets.
llm = GooglePalm(api_key=st.secrets["GOOGLE_API_KEY"])

# Function to convert business requirements into technical requirements using LLM.
def convert_business_to_technical(business_text):
    prompt_template = PromptTemplate(
        input_variables=["business_text"],
        template=(
            "Convert the following business requirements into detailed technical requirements "
            "including both functional and non-functional aspects:\n\n{business_text}"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    technical_requirements = chain.run(business_text=business_text)
    return technical_requirements

# Function to generate a Request for Proposal (RFP) document using technical requirements.
def generate_rfp(technical_requirements):
    prompt_template = PromptTemplate(
        input_variables=["technical_requirements"],
        template=(
            "Generate a professional Request for Proposal (RFP) document using the following "
            "technical requirements. Include clear performance criteria and all necessary technical details:\n\n{technical_requirements}"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    rfp_document = chain.run(technical_requirements=technical_requirements)
    return rfp_document

# Function to match vendors based on technical requirements using LLM.
def match_vendors(technical_requirements, vendor_df):
    # Convert a subset of vendor data to string for prompt brevity.
    vendor_data_str = vendor_df.head(10).to_csv(index=False)
    prompt_template = PromptTemplate(
         input_variables=["technical_requirements", "vendor_data"],
         template=(
             "Given the technical requirements:\n{technical_requirements}\n\nand the following vendor data:\n{vendor_data}\n\n"
             "Select the top 3 vendors best suited for the project based on historical performance and relevant expertise. "
             "Return the result in CSV format with columns: VendorName, KeyStrengths."
         )
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    output = chain.run(technical_requirements=technical_requirements, vendor_data=vendor_data_str)
    try:
         shortlisted = pd.read_csv(io.StringIO(output))
    except Exception as e:
         st.error("LLM output for vendor selection could not be parsed. Using fallback selection.")
         shortlisted = vendor_df.head(3)
    return shortlisted

# Function to evaluate bids using LLM.
def evaluate_bids(bids_df):
    # Convert a subset of bids data to string.
    bids_data_str = bids_df.head(10).to_csv(index=False)
    prompt_template = PromptTemplate(
         input_variables=["bids_data"],
         template=(
             "Evaluate the following bids and select the top 2 bids based on price, quality, delivery timelines, "
             "and technological capability. Return the result in CSV format with columns: BidID, EvaluationScore.\n\n{bids_data}"
         )
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    output = chain.run(bids_data=bids_data_str)
    try:
         evaluated = pd.read_csv(io.StringIO(output))
    except Exception as e:
         st.error("LLM output for bid evaluation could not be parsed. Using fallback evaluation.")
         evaluated = bids_df.head(2)
    return evaluated

# Function to simulate negotiation and generate a contract draft using LLM.
def simulate_negotiation_and_contract(top_bid):
    prompt_template = PromptTemplate(
         input_variables=["bid_details"],
         template=(
             "Based on the following top bid details:\n{bid_details}\n\n"
             "Generate a negotiation strategy and a draft contract. "
             "Separate the negotiation strategy and the contract draft with a delimiter '---'."
         )
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    bid_details = "\n".join([f"{key}: {value}" for key, value in top_bid.items()])
    output = chain.run(bid_details=bid_details)
    if "---" in output:
         negotiation_strategy, contract_draft = output.split("---", 1)
    else:
         negotiation_strategy = output
         contract_draft = "Contract draft could not be generated."
    return negotiation_strategy.strip(), contract_draft.strip()

# Initialize session state variables if not already set.
if 'business_requirements' not in st.session_state:
    st.session_state['business_requirements'] = ''
if 'technical_requirements' not in st.session_state:
    st.session_state['technical_requirements'] = ''
if 'rfp_document' not in st.session_state:
    st.session_state['rfp_document'] = ''
if 'vendor_df' not in st.session_state:
    st.session_state['vendor_df'] = None
if 'bids_df' not in st.session_state:
    st.session_state['bids_df'] = None
if 'shortlisted_vendors' not in st.session_state:
    st.session_state['shortlisted_vendors'] = None
if 'evaluated_bids' not in st.session_state:
    st.session_state['evaluated_bids'] = None
if 'negotiation_strategy' not in st.session_state:
    st.session_state['negotiation_strategy'] = ''
if 'contract_draft' not in st.session_state:
    st.session_state['contract_draft'] = ''

st.title("Procurement Automation Workflow")

# -------------------------------
# Step 1: Upload Inputs and Business Requirements
# -------------------------------
st.header("Step 1: Upload Inputs & Enter Business Requirements")
with st.form("input_form"):
    business_text = st.text_area("Enter Business Requirements", height=150)
    vendor_file = st.file_uploader("Upload Vendor History CSV", type=["csv"])
    bids_file = st.file_uploader("Upload Bids CSV", type=["csv"])
    submitted_inputs = st.form_submit_button("Submit Inputs")
    
    if submitted_inputs:
        # Capture business requirements.
        if business_text:
            st.session_state['business_requirements'] = business_text
            st.success("Business requirements captured.")
        else:
            st.error("Please enter business requirements.")
        
        # Process vendor CSV upload.
        if vendor_file is not None:
            try:
                vendor_df = pd.read_csv(vendor_file)
                st.session_state['vendor_df'] = vendor_df
                st.success("Vendor CSV uploaded successfully.")
            except Exception as e:
                st.error(f"Error reading vendor CSV: {e}")
        else:
            st.error("Please upload Vendor History CSV.")
        
        # Process bids CSV upload.
        if bids_file is not None:
            try:
                bids_df = pd.read_csv(bids_file)
                st.session_state['bids_df'] = bids_df
                st.success("Bids CSV uploaded successfully.")
            except Exception as e:
                st.error(f"Error reading bids CSV: {e}")
        else:
            st.error("Please upload Bids CSV.")

# -------------------------------
# Step 2: Convert Business Requirements to Technical Requirements
# -------------------------------
st.header("Step 2: Convert Business to Technical Requirements")
if st.session_state['business_requirements']:
    if st.button("Convert to Technical Requirements"):
        tech_req = convert_business_to_technical(st.session_state['business_requirements'])
        st.session_state['technical_requirements'] = tech_req
        st.write("Generated Technical Requirements:")
        st.text_area("Technical Requirements", value=tech_req, height=150)
else:
    st.info("Enter business requirements in Step 1.")

# -------------------------------
# Step 3: Generate RFP Document
# -------------------------------
st.header("Step 3: Generate Request for Proposal (RFP)")
if st.session_state['technical_requirements']:
    if st.button("Generate RFP"):
        rfp = generate_rfp(st.session_state['technical_requirements'])
        st.session_state['rfp_document'] = rfp
        st.write("Generated RFP Document:")
        st.text_area("RFP Document", value=rfp, height=150)
else:
    st.info("Please generate technical requirements in Step 2.")

# -------------------------------
# Step 4: Vendor Selection
# -------------------------------
st.header("Step 4: Vendor Selection")
if st.session_state['technical_requirements'] and st.session_state['vendor_df'] is not None:
    if st.button("Select Vendors"):
        shortlisted = match_vendors(st.session_state['technical_requirements'], st.session_state['vendor_df'])
        st.session_state['shortlisted_vendors'] = shortlisted
        st.write("Shortlisted Vendors:")
        st.dataframe(shortlisted)
else:
    st.info("Ensure technical requirements are generated and vendor CSV is uploaded.")

# -------------------------------
# Step 5: Evaluate Bids
# -------------------------------
st.header("Step 5: Bid Evaluation")
if st.session_state['bids_df'] is not None:
    if st.button("Evaluate Bids"):
        evaluated = evaluate_bids(st.session_state['bids_df'])
        st.session_state['evaluated_bids'] = evaluated
        st.write("Top Evaluated Bids:")
        st.dataframe(evaluated)
else:
    st.info("Please upload Bids CSV in Step 1.")

# -------------------------------
# Step 6: Negotiation Simulation & Contract Drafting
# -------------------------------
st.header("Step 6: Negotiation Simulation and Contract Drafting")
if st.session_state['evaluated_bids'] is not None and not st.session_state['evaluated_bids'].empty:
    # For demonstration, we take the top bid from the evaluated bids.
    top_bid = st.session_state['evaluated_bids'].iloc[0].to_dict()
    if st.button("Simulate Negotiation & Draft Contract"):
        negotiation_strategy, contract_draft = simulate_negotiation_and_contract(top_bid)
        st.session_state['negotiation_strategy'] = negotiation_strategy
        st.session_state['contract_draft'] = contract_draft
        st.write("Negotiation Strategy:")
        st.text_area("Negotiation Strategy", value=negotiation_strategy, height=100)
        st.write("Contract Draft:")
        st.text_area("Contract Draft", value=contract_draft, height=150)
else:
    st.info("Please evaluate bids in Step 5.")

# -------------------------------
# Step 7: Final Review & Download Options
# -------------------------------
st.header("Step 7: Final Review & Download")
if st.session_state['rfp_document']:
    st.subheader("RFP Document")
    st.text_area("RFP Document", value=st.session_state['rfp_document'], height=150)
if st.session_state['technical_requirements']:
    st.subheader("Technical Requirements")
    st.text_area("Technical Requirements", value=st.session_state['technical_requirements'], height=150)
if st.session_state['shortlisted_vendors'] is not None:
    st.subheader("Shortlisted Vendors")
    st.dataframe(st.session_state['shortlisted_vendors'])
if st.session_state['evaluated_bids'] is not None:
    st.subheader("Evaluated Bids")
    st.dataframe(st.session_state['evaluated_bids'])
if st.session_state['negotiation_strategy']:
    st.subheader("Negotiation Strategy")
    st.text_area("Negotiation Strategy", value=st.session_state['negotiation_strategy'], height=100)
if st.session_state['contract_draft']:
    st.subheader("Contract Draft")
    st.text_area("Contract Draft", value=st.session_state['contract_draft'], height=150)

# Download options for final documents.
st.header("Download Final Documents")
if st.session_state['rfp_document']:
    st.download_button("Download RFP Document", st.session_state['rfp_document'], file_name="RFP_Document.txt")
if st.session_state['technical_requirements']:
    st.download_button("Download Technical Requirements", st.session_state['technical_requirements'], file_name="Technical_Requirements.txt")
if st.session_state['contract_draft']:
    st.download_button("Download Contract Draft", st.session_state['contract_draft'], file_name="Contract_Draft.txt")
