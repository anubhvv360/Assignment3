import streamlit as st
import langchain
import pandas as pd
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title = "Procurement Agent", page_icon = "📦")
st.title("Transglobal Procurement Agent")

# displaying versions of libraries used
print("streamlit version:", st.__version__)
print("langchain version:", langchain.__version__)
print("pandas version:", pd.__version__)

# -------------------------------
# 1. Initialize the Chat Model
# -------------------------------
# We'll load the LLM in a cached resource function so it's not re-initialized on each run.
@st.cache_resource
def load_llm():
    """
    Initialize the ChatGoogleGenerativeAI model with your chosen parameters.
    Make sure you have your GOOGLE_API_KEY in st.secrets["GOOGLE_API_KEY"].
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",  # or another available model (e.g. "chat-bison@001")
        temperature=0,
        max_tokens=8000,
        api_key=st.secrets["GOOGLE_API_KEY"]
    )

llm = load_llm()

# -------------------------------
# 2. LLM-Driven Functions
# -------------------------------
def convert_business_to_technical(business_text):
    """Use the LLM to convert business requirements into technical requirements."""
    prompt_template = """You are a Senior Technical Procurement Specialist at a manufacturing firm. Your task is to convert the following Business Requirements Document (BRD) into a comprehensive, strictly technical requirements document. Use only the details provided in the BRD and do not add any external information. The output must include both functional and non-functional requirements for purchasing new servers, software, or other technical assets, and it should be written so that suppliers can unambiguously understand every specification.
                        Your output must clearly address each of the following key factors and include specific examples or ranges where applicable:
                        Requirement Clarity & Specification Accuracy
                        Example: If the BRD specifies intensive software handling, include a statement like "System must include 16 GB DDR4 RAM or more" to support heavy multitasking.
                        Example: For high computational performance, specify a processor such as "Intel Core i7 (11th Gen) or equivalent" or higher.
                        Standardized Descriptions
                        Example: Instead of vague terms like "fast processor," clearly state "Processor: Intel Core i7 (11th Gen) or better."
                        Example: For storage, specify "512GB SSD or higher" if the BRD indicates high storage needs.
                        Hidden Costs Consideration
                        Example: Include notes like "Include considerations for logistics, customs, and storage costs" when the BRD mentions any potential hidden costs.
                        Example: If delivery or installation is mentioned, ensure to note any additional expenses that might be incurred.
                        Regulatory Compliance
                        Example: If the BRD refers to adhering to industry standards, include "Ensure compliance with import/export laws and applicable industry standards."
                        Example: Specify required certifications or licenses if mentioned (e.g., "Operating System must be a licensed version of Windows 11 Pro").
                        Risk Mitigation
                        Example: If the BRD highlights potential risks, output "Include contract clauses for warranties (e.g., 3 years onsite warranty), penalties for delays, and risk mitigation measures."
                        Example: Mention hidden risks like "Consideration for potential downtime and its impact on operations."
                        Additionally, use the following example vendor bid to guide the level of detail expected. This example is for reference only and should not be copied verbatim unless the BRD contains matching details:
                        Processor: Intel Core i7 (11th Gen) or equivalent
                        RAM: 16 GB DDR4 or more
                        Storage: 512GB SSD or higher
                        Display: 15.6-inch FHD (1920×1080) IPS
                        Graphics: Integrated or discrete as specified
                        Battery Life: 10 hours or more
                        Ports: e.g., 2× USB 3.0, 1× USB Type-C, HDMI, 3.5mm audio jack, Ethernet
                        Connectivity: Wi-Fi 6, Bluetooth 5.0
                        Operating System: Windows 11 Pro (licensed)
                        Keyboard: Backlit keyboard
                        Weight: Approximately 2.2 kg
                        Warranty: 3 years onsite warranty
                        Software: Pre-installed Windows 11 Pro with license, Office 365 Business, Antivirus with a 3-year subscription
                        Financial Proposal: e.g., Unit Price: $1,200 per unit; total cost for 100 units: $120,000; additional software/services: $5,000; Payment Terms: 50% advance, 50% on delivery
                        Instruction:
                        Convert the provided BRD into a detailed Technical Requirements Document that strictly contains technical details extracted from the BRD. 
                        For each key technical aspect (e.g., processor, RAM, storage), if the BRD suggests intensive or high-performance needs, the output must specify exact or minimum values such as "16 GB DDR4" or "512GB SSD or more" as appropriate. Similarly, include specific requirements for connectivity, operating system, software, hidden cost considerations, regulatory compliance, and risk mitigation measures.
                        BRD: {business_text}"""
    
    prompt = PromptTemplate(input_variables=["business_text"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(business_text=business_text)

def generate_rfp(technical_requirements):
    """Use the LLM to generate an RFP document from technical requirements."""
    prompt_template = """Convert the following Technical Requirements Document into a comprehensive and professional Request for Proposal (RFP) document.
                      The RFP should clearly articulate all technical details and performance criteria required from potential suppliers, based solely on the provided input.
                      Technical Requirements: {technical_requirements}
                      NOTE: do not introduce any external details or hallucinations."""
    
    prompt = PromptTemplate(input_variables=["technical_requirements"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(technical_requirements=technical_requirements)

def match_vendors(vendor_df):
    """
    Use the LLM to select top vendors based on rpf and vendor data.
    For simplicity, we only pass a subset of vendor data to the prompt.
    """
    # Add technical requirements from rfp and vendor evaluation criteria to the prompt
    # Convert the vendor DataFrame to a CSV string
    vendor_csv_str = vendor_df.to_csv(index=False)
    # Acceptance criteria: Quality_of_Goods: 0.4, Delivery_punctuality: 0.35,  Contract_term_compliance: 0.25
    prompt_template =  """You are provided with a vendor dataset as input (variable: {vendor_data}) where each row represents a single delivery. The dataset includes the following key attributes:

                        Vendor_name: Name of the vendor.
                        Delivery_punctuality: Delivery punctuality score (1-10, with 10 being the best).
                        Quality_of_goods: Quality score for delivered goods (1-10, with 10 being the best).
                        Contract_term_compliance: Contract compliance score (1-10, with 10 being the best).
                        
                        Follow these steps:
                        
                        1. Aggregate Data: Group the dataset by 'Vendor_name'. For each vendor, calculate the mean values for 'Delivery_punctuality', 'Quality_of_goods', and 'Contract_term_compliance'. For example, for Vendor A, determine:
                        
                        Mean_Delivery_punctuality_Vendor_A
                        Mean_Quality_of_goods_Vendor_A
                        Mean_Contract_term_compliance_Vendor_A
                        
                        2. Compute Weighted Average: For each vendor, compute the weighted average score using the weights:
                        
                        w1 = 0.4 for 'Quality_of_goods'
                        w2 = 0.35 for 'Delivery_punctuality'
                        w3 = 0.25 for 'Contract_term_compliance'
                        Specifically, for a vendor X:
                        Weighted_Average_X = (Mean_Quality_of_goods_X * 0.4) + (Mean_Delivery_punctuality_X * 0.35) + (Mean_Contract_term_compliance_X * 0.25)
                        
                        3. Select Top Vendors: Rank the vendors based on their weighted average scores in descending order and choose the top 3.
                        
                        4. Output Format: Return a list of the top 3 vendors.
                        
                        Remember, Use only the information provided in the vendor dataset. Do not introduce any external data or hallucinate details.
                        Do not return python code.
                        Output Format:
                        Strictly return only the CSV output with column: VendorName
                        Do not return any other strings or text"""
    
    prompt = PromptTemplate(input_variables=["vendor_data"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(vendor_data=vendor_csv_str)
    # Clean up output to remove extra blank lines or unwanted text
    output = output.strip()
    # try:
    shortlisted = pd.read_csv(io.StringIO(output))
    # except Exception:
    #     st.error("Could not parse LLM output for vendor selection. Using fallback selection.")
    #     shortlisted = vendor_df.head(3)
    return shortlisted

def generate_tender_doc(tech_req):
    prompt_template = """ Using the provided Technical Requirements Document (TRD) (variable: {trd}), generate a professional tender document for procurement purposes. The tender document should be structured, clear, and concise, ensuring that vendors fully understand the technical and business requirements.
                        The tender document should include the following sections:
                        
                        1.	Title & Issuing Organization: Clearly state the tender title and the organization issuing it.
                        2.	Invitation to Tender: A brief introduction explaining the purpose of the tender and the procurement scope.
                        3.	Scope of Work: A summary of vendor responsibilities, including product delivery, pre-installation requirements, and support expectations.
                        4.	Technical Requirements: 
                            o	Hardware specifications (processor, RAM, storage, display, connectivity, etc.).
                            o	Software requirements (pre-installed OS, applications, security software).
                            o	Security and compliance requirements (e.g., TPM, biometric authentication).
                        5.	Business Requirements: 
                            o	Quantity of units required.
                            o	Pricing and expected bulk purchase discounts.
                            o	Payment terms and preferred conditions.
                            o	Warranty and support expectations.
                            o	Delivery timelines and shipping requirements.
                            o	Packaging and inventory tagging needs.
                        6.	Proposal Submission Guidelines: 
                            o	Instructions on submission format (e.g., PDF).
                            o	Required documents (e.g., company profile, compliance statement, pricing breakdown).
                            o	Submission deadline and contact details.
                        7.	Closing Statement: A formal closing encouraging vendors to submit proposals and reinforcing the deadline.
                        
                        Ensure that the document is professionally formatted with section headers, bullet points where necessary, and a business-appropriate tone. The final output should be structured in a way that makes it easy for vendors to review and respond efficiently.
                        Remember, Use only the information provided in the TRD. Do not introduce any external data or hallucinate details."""
    
    prompt = PromptTemplate(input_variables=["trd"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(trd = tech_req)

def generate_email(rfp):
    prompt_template = """ Generate a professional email addressed to the shortlisted vendors inviting them to submit their bids for the attached Tender Document and Request for Proposal (RFP). The email should be formal, clear, and concise, ensuring that vendors understand the expectations and submission requirements.
                        Check for the name of Issuing organization from the attached RFP document (Variable: {rfp})
                        The email should include only the following elements:
                        1.	Subject Line: A professional subject line indicating an invitation to submit a bid.
                        2.	Salutation: Address the vendors respectfully.
                        3.	Introduction: Briefly introduce the issuing organization and the purpose of the email.
                        4.	Invitation to Bid: Clearly state that the vendor has been shortlisted and is invited to submit a proposal based on the attached documents.
                        5.	Attachments: Mention the attached Tender Document and RFP for their reference.
                        6.	Closing Statement: Encourage timely submissions and express anticipation for their response.
                        7.	Signature: Include the sender’s name, designation, and contact details.
                        Ensure the email is concise, professional, and engaging, while maintaining a clear call to action for the vendors to submit their bids on time."""
  
    prompt = PromptTemplate(input_variables=["rfp"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(rfp = rfp)

def evaluate_bids(bids_df):
    """
    Use the LLM to evaluate bids and pick the top 2 based on price, quality, timelines, etc.
    Again, we only pass a sample of the bids to keep the prompt short.
    """
    # bids_data_str = bids_df.head(10).to_csv(index=False)
    prompt_template = """You have the following bids:\n{bids_df}\n\n
                      Evaluate each bid based on price, quality, delivery timelines, and technology.
                      Select the top 2 bids and return them in CSV format with columns: BidID, EvaluationScore."""

    prompt = PromptTemplate(input_variables=["bids_df"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(bids_data=bids_data_str)
    try:
        evaluated = pd.read_csv(io.StringIO(output))
    except Exception:
        st.error("Could not parse LLM output for bid evaluation. Using fallback.")
        evaluated = bids_df.head(2)
    return evaluated

def simulate_negotiation_and_contract(top_bid):
    """
    Use the LLM to simulate a negotiation strategy and generate a contract draft from the top bid.
    """
    bid_details = "\n".join([f"{k}: {v}" for k, v in top_bid.items()])
    prompt_template = """You are a Procurement Negotiator with top 2 bid details:\n{bid_details}\n\n

                        First, outline a robust negotiation strategy. Then, draft a contract reflecting your strategy. Separate the negotiation strategy and the draft contract with '---'.
                        
                        Analyze the bids' data to determine the company's Best Alternative to a Negotiated Agreement (BATNA). Then, using LLM-driven insights, simulate negotiation scenarios to devise robust negotiation strategies for engaging with the preferred supplier. 
                        Your recommendations should ensure that the procuring company is well-prepared to secure favorable terms by leveraging competitive market trends, supplier pricing, and potential bulk discounts.

                        Using first principles thinking, break down the negotiation challenge into its fundamental components. Identify the core drivers—such as supplier cost structures, market trends, and value determinants—without relying on conventional assumptions. 
                        Reconstruct a negotiation strategy from these basic truths that challenges standard practices and leverages competitive insights.

                        Apart from other vital things, include the key factors below also:
                        BATNA: Evaluate alternatives, given the two bids.
                        Market Trends & Supplier Pricing: Analyze current trends and pricing.
                        Benchmarking: Compare prices across vendors.
                        Bulk Discounts: Assess potential savings on volume purchases.
                        Leverage Competition: Use the competitive environment to negotiate better terms."""

    prompt = PromptTemplate(input_variables=["bid_details"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(bid_details=bid_details)
    if "---" in output:
        negotiation_strategy, contract_draft = output.split("---", 1)
    else:
        negotiation_strategy = output
        contract_draft = "No contract draft found."
    return negotiation_strategy.strip(), contract_draft.strip()

# -------------------------------
# 3. Initialize Session State
# -------------------------------
# if 'disabled' not in st.session_state:
#     st.session_state.disabled = False
if 'business_requirements' not in st.session_state:
    st.session_state['business_requirements'] = ''
if 'technical_requirements' not in st.session_state:
    st.session_state['technical_requirements'] = ''
if 'rfp_document' not in st.session_state:
    st.session_state['rfp_document'] = ''
if 'vendor_df' not in st.session_state:
    st.session_state['vendor_df'] = None
if 'tender_doc' not in st.session_state:
    st.session_state['tender_doc'] = ''
if 'email' not in st.session_state:
    st.session_state['email'] = ''
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

# -------------------------------
# 4. Streamlit App Layout
# -------------------------------


# Step 1: Inputs
st.header("Step 1: Upload Inputs & Business Requirements")
with st.form("input_form"):
    business_text = st.text_area("Enter Business Requirements", height=150)
    # vendor_file = st.file_uploader("Upload Vendor History CSV", type=["csv"])
    # bids_file = st.file_uploader("Upload Bids CSV", type=["csv"])
    submitted_inputs = st.form_submit_button("Submit Inputs")

    if submitted_inputs:
        # Capture business requirements
        if business_text:
            st.session_state['business_requirements'] = business_text
            st.success("Business requirements captured.")
        else:
            st.error("Please enter business requirements.")
        
        # Process vendor CSV
        # if vendor_file is not None:
        #     try:
        #         vendor_df = pd.read_csv(vendor_file)
        #         st.session_state['vendor_df'] = vendor_df
        #         st.success("Vendor CSV uploaded successfully.")
        #     except Exception as e:
        #         st.error(f"Error reading vendor CSV: {e}")
        # else:
        #     st.error("Please upload Vendor History CSV.")
        
        # # Process bids CSV
        # if bids_file is not None:
        #     try:
        #         bids_df = pd.read_csv(bids_file)
        #         st.session_state['bids_df'] = bids_df
        #         st.success("Bids CSV uploaded successfully.")
        #     except Exception as e:
        #         st.error(f"Error reading Bids CSV: {e}")
        # else:
        #     st.error("Please upload Bids CSV.")

# Step 2: Convert to Technical Requirements
st.header("Step 2: Convert Business to Technical Requirements")
if st.session_state['business_requirements']:
    if st.button("Convert to Technical Requirements"):
        tech_req = convert_business_to_technical(st.session_state['business_requirements'])
        st.session_state['technical_requirements'] = tech_req
        st.success("Generated Technical Requirements")
        with st.expander("Show Technical Requirements"):
            st.write(tech_req)
        # st.write("Generated Technical Requirements:")
        # st.text_area("Technical Requirements", value=tech_req, height=150)
else:
    st.info("Enter business requirements in Step 1.")

# Step 3: Generate RFP
st.header("Step 3: Generate RFP")
if st.session_state['technical_requirements']:
    if st.button("Generate RFP"):
        rfp = generate_rfp(st.session_state['technical_requirements'])
        st.session_state['rfp_document'] = rfp
        st.success("Generated RFP")
        with st.expander("Show RFP"):
            st.write(rfp)
        # st.text_area("RFP Document", value=rfp, height=150)
else:
    st.info("Please generate technical requirements in Step 2.")

# Step 4: Vendor Selection
st.header("Step 4: Vendor Selection")
if st.session_state['rfp_document']:
    vendor_file = st.file_uploader("Upload Vendor History CSV", type=["csv"])
    if st.button("Select Vendors"):
        if vendor_file is not None:
            try:
                vendor_df = pd.read_csv(vendor_file)
                st.session_state['vendor_df'] = vendor_df
                st.success("Vendor CSV uploaded successfully.")
                shortlisted = match_vendors(st.session_state['vendor_df'])
                st.session_state['shortlisted_vendors'] = shortlisted
                st.success("Shortlisted Vendors")
                with st.expander("Show shortlisted vendors"):
                    st.dataframe(shortlisted)
            except Exception as e:
                st.error(f"Error reading vendor CSV: {e}")
                
        else:
            st.error("Please upload Vendor History CSV.")        
else:
    st.info("Ensure technical requirements are generated.")

# Step 5: Producing a tender document and generating email for the shortlisted vendors
st.header("Step 5: Generating Emails for vendors")
if st.session_state['shortlisted_vendors'] is not None:
    if st.button("Generate Tender Document"):
        tender_doc = generate_tender_doc(st.session_state['technical_requirements'])
        st.session_state['tender_doc'] = tender_doc
        st.success("Generated Tender Document")
        with st.expander("Show Tender Document"):
            st.write(tender_doc)
   
    if st.session_state['tender_doc']:
        if st.button("Generate Email for shortlisted Vendors"):
            email = generate_email(st.session_state['rfp_document'])
            st.session_state['email'] = email
            st.success("Generated Email for vendors")
            with st.expander("Show Email"):
                st.write(email)
    else:
        st.info("Ensure Tender Document is generated")
else:
    st.info("Ensure shortlisted vendors list is generated")


# Step 6: Bid Evaluation
st.header("Step 6: Evaluate Bids")
# Replace the below if statement with "if tender document has been generated or not in the Step 4"
if st.session_state['shortlisted_vendors'] is not None:
    bids_file = st.file_uploader("Upload Bids CSV", type=["csv"])
    if st.button("Evaluate Bids"):
        evaluated = evaluate_bids(st.session_state['bids_df'])
        st.session_state['evaluated_bids'] = evaluated
        st.success("Evaluated Bids")
        with st.expander("Show Top Evaluated Bids"):
            st.dataframe(evaluated)
    
# if st.session_state['bids_df'] is not None:
#     if st.button("Evaluate Bids"):
#         evaluated = evaluate_bids(st.session_state['bids_df'])
#         st.session_state['evaluated_bids'] = evaluated
#         st.success("Evaluated Bids")
#         with st.expander("Show Top Evaluated Bids"):
#             st.dataframe(evaluated)
        # st.write("Top Evaluated Bids:")
        # st.dataframe(evaluated)
else:
    st.info("Please upload Bids CSV in Step 1.")

# Step 7: Negotiation & Contract
st.header("Step 7: Negotiation Simulation and Contract Drafting")
if st.session_state['evaluated_bids'] is not None and not st.session_state['evaluated_bids'].empty:
    top_bid = st.session_state['evaluated_bids'].iloc[0].to_dict()
    if st.button("Simulate Negotiation & Draft Contract"):
        negotiation_strategy, contract_draft = simulate_negotiation_and_contract(top_bid)
        st.session_state['negotiation_strategy'] = negotiation_strategy
        st.session_state['contract_draft'] = contract_draft
        st.success("Generated Negotiation Strategy and Contract Draft")
        with st.expander("Show Negotiation Strategy"):
            st.write(negotiation_strategy)
        # st.write("Negotiation Strategy:")
        # st.text_area("Negotiation Strategy", value=negotiation_strategy, height=100)
        with st.expander("Show Contract Draft"):
            st.write(contract_draft)
        # st.write("Contract Draft:")
        # st.text_area("Contract Draft", value=contract_draft, height=150)
else:
    st.info("Please evaluate bids in Step 5.")

# Step 8: Final Review & Downloads
st.header("Step 8: Final Review & Download")
if st.session_state['technical_requirements']:
    with st.expander("Show Technical Requirements"):
        st.write(st.session_state['technical_requirements'])

if st.session_state['rfp_document']:
    with st.expander("Show Request For Proposal"):
        st.write(st.session_state['rfp_document'])

if st.session_state['shortlisted_vendors'] is not None:
    with st.expander("Show shortlisted vendors"):
        st.dataframe(st.session_state['shortlisted_vendors'])

if st.session_state['tender_doc']:
    with st.expander("Show Tender Document"):
        st.write(st.session_state['tender_doc'])
    
if st.session_state['evaluated_bids'] is not None:
    with st.expander("Show Top Evaluated Bids"):
        st.dataframe(st.session_state['evaluated_bids'])

if st.session_state['negotiation_strategy']:
    with st.expander("Show Negotiation Strategy"):
        st.write(st.session_state['negotiation_strategy'])

if st.session_state['contract_draft']:
    with st.expander("Show Contract Draft"):
        st.write(st.session_state['contract_draft'])

st.header("Download Final Documents")
if st.session_state['rfp_document']:
    st.download_button("Download RFP Document", st.session_state['rfp_document'], file_name="RFP_Document.txt")
if st.session_state['technical_requirements']:
    st.download_button("Download Technical Requirements", st.session_state['technical_requirements'], file_name="Technical_Requirements.txt")
if st.session_state['contract_draft']:
    st.download_button("Download Contract Draft", st.session_state['contract_draft'], file_name="Contract_Draft.txt")
