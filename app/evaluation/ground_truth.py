from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class GroundTruthItem:
    question_id: str
    question: str
    expected_pages: List[int]
    expected_keyphrases: List[str]
    expected_answer: str

# Ground Truth Benchmark Dataset for Shibpur Tender (Tendernotice_1.pdf + BOQ_966022.xls)
SHIBPUR_GROUND_TRUTH: List[GroundTruthItem] = [
    GroundTruthItem(
        question_id="Q1",
        question="What is the name or title of the work?",
        expected_pages=[1],
        expected_keyphrases=["Repair and renovation of rooms at SOMS", "Repair and renovation"],
        expected_answer="Repair and renovation of rooms at SOMS"
    ),
    GroundTruthItem(
        question_id="Q2",
        question="What is the tender enquiry number or NIT reference number?",
        expected_pages=[1],
        expected_keyphrases=["e-Proc/CWSOMS_27072026/PD _11EST/668R", "CWSOMS", "668R"],
        expected_answer="e-Proc/CWSOMS_27072026/PD _11EST/668R"
    ),
    GroundTruthItem(
        question_id="Q3",
        question="Which institute, department, or organization issued this tender?",
        expected_pages=[1, 2],
        expected_keyphrases=["IIEST", "Shibpur", "Indian Institute of Engineering Science and Technology"],
        expected_answer="Indian Institute of Engineering Science and Technology (IIEST), Shibpur"
    ),
    GroundTruthItem(
        question_id="Q4",
        question="What is the location or site of the proposed work?",
        expected_pages=[1],
        expected_keyphrases=["SOMS", "Shibpur"],
        expected_answer="SOMS, IIEST Shibpur campus"
    ),
    GroundTruthItem(
        question_id="Q5",
        question="What is the tender type or mode of bidding?",
        expected_pages=[1],
        expected_keyphrases=["e-Tenders", "online", "Central Public Procurement Portal"],
        expected_answer="Online e-Tender via CPP Portal"
    ),
    GroundTruthItem(
        question_id="Q6",
        question="What is the estimated cost of the tender?",
        expected_pages=[1, 2],
        expected_keyphrases=["estimated cost"],
        expected_answer="Not specified in the provided document"
    ),
    GroundTruthItem(
        question_id="Q7",
        question="What is the EMD (Earnest Money Deposit) amount or percentage?",
        expected_pages=[1],
        expected_keyphrases=["1%", "estimated cost", "EMD"],
        expected_answer="1% of the estimated cost"
    ),
    GroundTruthItem(
        question_id="Q8",
        question="What are the acceptable modes or forms of EMD payment?",
        expected_pages=[1],
        expected_keyphrases=["demand draft", "The Registrar, IIEST, Shibpur"],
        expected_answer="Demand Draft in favor of 'The Registrar, IIEST, Shibpur' payable at Kolkata"
    ),
    GroundTruthItem(
        question_id="Q9",
        question="To whom should the EMD demand draft be drawn in favor of?",
        expected_pages=[1],
        expected_keyphrases=["The Registrar, IIEST, Shibpur"],
        expected_answer="The Registrar, IIEST, Shibpur"
    ),
    GroundTruthItem(
        question_id="Q10",
        question="Where and by when must the hard copy of EMD be submitted?",
        expected_pages=[1],
        expected_keyphrases=["e-Procurement Cell", "Registrar Office", "16.08.2026"],
        expected_answer="e-Procurement Cell, 1st Floor, Administration Building, IIEST Shibpur by 16.08.2026 at 12:00 pm"
    ),
    GroundTruthItem(
        question_id="Q11",
        question="Is Micro and Small Enterprises (MSE) or MSME EMD exemption allowed?",
        expected_pages=[1],
        expected_keyphrases=["Exemption will be provided", "valid documents"],
        expected_answer="Yes, EMD exemption is provided subject to valid document submission"
    ),
    GroundTruthItem(
        question_id="Q12",
        question="What is the required Performance Security or Performance Guarantee percentage?",
        expected_pages=[1],
        expected_keyphrases=["5%", "Performance Security", "Contract value"],
        expected_answer="5% of total ordered/contract value"
    ),
    GroundTruthItem(
        question_id="Q13",
        question="What is the Security Deposit percentage to be deducted from bills?",
        expected_pages=[1, 2],
        expected_keyphrases=["security deposit", "performance security"],
        expected_answer="Included in performance security deposit"
    ),
    GroundTruthItem(
        question_id="Q14",
        question="What is the validity period of the performance security?",
        expected_pages=[1],
        expected_keyphrases=["performance security"],
        expected_answer="Valid through contract completion plus defect liability period"
    ),
    GroundTruthItem(
        question_id="Q15",
        question="Is tender document fee or cost applicable?",
        expected_pages=[1],
        expected_keyphrases=["tender"],
        expected_answer="Not specified in document"
    ),
    GroundTruthItem(
        question_id="Q16",
        question="What is the last date and time for online bid submission?",
        expected_pages=[1],
        expected_keyphrases=["16.08.2026", "12:00 pm"],
        expected_answer="16.08.2026 at 12:00 pm"
    ),
    GroundTruthItem(
        question_id="Q17",
        question="When and at what time will the technical/eligibility bid be opened?",
        expected_pages=[1],
        expected_keyphrases=["17.08.2026", "12:30 pm", "Opening of Bid"],
        expected_answer="17.08.2026 at 12:30 pm"
    ),
    GroundTruthItem(
        question_id="Q18",
        question="When will the financial bid be opened?",
        expected_pages=[1],
        expected_keyphrases=["Opening of Bid", "17.08.2026"],
        expected_answer="Date of opening of financial bid is not separately specified"
    ),
    GroundTruthItem(
        question_id="Q19",
        question="What is the bid validity period in days?",
        expected_pages=[1],
        expected_keyphrases=["90", "ninety", "days"],
        expected_answer="Not less than 90 (ninety) days"
    ),
    GroundTruthItem(
        question_id="Q20",
        question="What is the period of completion or execution time allowed for the work?",
        expected_pages=[3],
        expected_keyphrases=["30 days"],
        expected_answer="30 days"
    ),
    GroundTruthItem(
        question_id="Q26",
        question="What is the required experience for similar completed works in the last 7 years?",
        expected_pages=[1],
        expected_keyphrases=["last 7 years", "30%", "40%", "50%", "80%"],
        expected_answer="3 works costing >= 40%, 2 works >= 50%, or 1 work >= 80% of estimated cost"
    ),
    GroundTruthItem(
        question_id="Q27",
        question="What is the cost threshold for 3 similar completed works?",
        expected_pages=[1],
        expected_keyphrases=["40%", "estimated cost"],
        expected_answer="40% of estimated cost"
    ),
    GroundTruthItem(
        question_id="Q28",
        question="What is the cost threshold for 2 similar completed works?",
        expected_pages=[1],
        expected_keyphrases=["50%", "estimated cost"],
        expected_answer="50% of estimated cost"
    ),
    GroundTruthItem(
        question_id="Q29",
        question="What is the cost threshold for 1 similar completed work?",
        expected_pages=[1],
        expected_keyphrases=["80%", "estimated cost"],
        expected_answer="80% of estimated cost"
    ),
    GroundTruthItem(
        question_id="Q30",
        question="What is the average annual financial turnover requirement?",
        expected_pages=[2],
        expected_keyphrases=["30%", "last 3 years"],
        expected_answer="30% of estimated cost"
    ),
    GroundTruthItem(
        question_id="Q32",
        question="Is Joint Venture (JV) or Consortium allowed in this tender?",
        expected_pages=[2],
        expected_keyphrases=["joint ventures", "not accepted"],
        expected_answer="No, joint ventures / consortium are not accepted"
    ),
    GroundTruthItem(
        question_id="Q36",
        question="Is GST registration certificate required?",
        expected_pages=[3, 4],
        expected_keyphrases=["GST", "registered under GST"],
        expected_answer="Yes, valid GST registration certificate is mandatory"
    ),
    GroundTruthItem(
        question_id="Q37",
        question="Is Permanent Account Number (PAN) card required?",
        expected_pages=[4],
        expected_keyphrases=["PAN", "Permanent Account Number"],
        expected_answer="Yes, copy of PAN card is required"
    ),
    GroundTruthItem(
        question_id="Q46",
        question="What is the Period of Maintenance or Defect Liability Period (DLP)?",
        expected_pages=[2],
        expected_keyphrases=["Period of Maintenance", "twelve months"],
        expected_answer="12 months (twelve months) from taking over possession"
    ),
    GroundTruthItem(
        question_id="Q56",
        question="Is mobilization advance or any advance payment allowed?",
        expected_pages=[2],
        expected_keyphrases=["No advance payment", "under any circumstance"],
        expected_answer="No advance payment will be made under any circumstance"
    ),
    GroundTruthItem(
        question_id="Q66",
        question="Who is the contact person for site inspection or queries?",
        expected_pages=[1],
        expected_keyphrases=["Mr. Dibyendu Banerjee", "9434114888"],
        expected_answer="Mr. Dibyendu Banerjee, Assistant Engg. (Civil), Mobile: 9434114888"
    ),
    GroundTruthItem(
        question_id="Q67",
        question="What is the phone or mobile number of the contact person?",
        expected_pages=[1],
        expected_keyphrases=["9434114888"],
        expected_answer="9434114888"
    ),
    GroundTruthItem(
        question_id="Q68",
        question="What is the official email address for sending tender queries?",
        expected_pages=[1],
        expected_keyphrases=["dibban2003@yahoo.co.in", "dibban2003"],
        expected_answer="dibban2003@yahoo.co.in"
    )
]
