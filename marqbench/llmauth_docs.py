"""
llmauth_docs.py — Published data dictionaries for MARQ-Bench condition A3.

Condition A3 supplies the rule author with the documentation a practitioner
would actually have: the field descriptions published alongside each corpus.

WHY THESE ARE NOT SANITISED
---------------------------
Every one of these corpora documents its own encoding conventions. UCI states
that `pdays = -1` means the client was not previously contacted; TLC states
that `RatecodeID = 99` means Null/unknown; the diabetes documentation states
that `max_glu_serum = "None"` means the test was not measured.

That is the entire point of A3. The condition tests whether supplying published
documentation is *sufficient* for an author to avoid the sentinel trap. Removing
those clauses would destroy the condition and turn A3 into an expensive copy of
A2. If models still misread sentinels with the documentation in hand, that is a
stronger and more uncomfortable finding than "models need profiling".

Descriptions are paraphrased from the published sources, not copied verbatim.

ANTI-TAUTOLOGY BOUNDARY
-----------------------
Authoring side. Imports nothing. Contains no failure codes, no sentinel
registry, and no statement about what an author *should* do — only what each
field means.

Author: Ramesh Babu Kallam
License: MIT
"""

from __future__ import annotations

__all__ = ["DATA_DICTIONARIES", "TABLE_DESCRIPTIONS", "get_docs", "get_table_doc",
           "coverage_report", "DOCS_VERSION"]

DOCS_VERSION = "1.0.0"


TABLE_DESCRIPTIONS: dict[str, str] = {
    "bank_marketing":
        "Direct marketing campaign records from a Portuguese retail bank. Each "
        "row is one telephone contact with a client. The campaign goal was to "
        "sell a term deposit product.",
    "diabetes_130us":
        "Ten years of clinical care records (1999-2008) from 130 US hospitals "
        "and integrated delivery networks. Each row is one inpatient encounter "
        "for a patient with diabetes, covering admission, labs, medications, "
        "and readmission outcome.",
    "online_retail_ii":
        "Transactional records from a UK-based online retailer selling "
        "giftware, covering 2009 to 2011. Each row is one line item on an "
        "invoice. Many customers are wholesalers.",
    "nyc_tlc_yellow":
        "Yellow taxi trip records from New York City, submitted by "
        "technology providers under the Taxicab Passenger Enhancement "
        "Program. Each row is one trip, with pickup and dropoff times and "
        "zones, distance, itemised fares, and payment type. Records are "
        "captured by vendors and were not verified by the TLC.",
}


_MEDICATION_DOC = (
    "Diabetic medication feature. Indicates whether the drug was prescribed or "
    "the dosage changed during the encounter: 'up' if increased, 'down' if "
    "decreased, 'steady' if unchanged, and 'No' if the drug was not prescribed."
)

_MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


DATA_DICTIONARIES: dict[str, dict[str, str]] = {

    # ---------------------------------------------------------------- C1
    "bank_marketing": {
        "age": "Client age in years (numeric).",
        "job": "Type of job (categorical: admin., unknown, unemployed, "
               "management, housemaid, entrepreneur, student, blue-collar, "
               "self-employed, retired, technician, services).",
        "marital": "Marital status (categorical: married, divorced, single). "
                   "'divorced' covers both divorced and widowed.",
        "education": "Education level (categorical: unknown, primary, "
                     "secondary, tertiary).",
        "default": "Whether the client has credit in default (binary: yes, no).",
        "balance": "Average yearly account balance in euros (numeric). May be "
                   "negative where the account is overdrawn.",
        "housing": "Whether the client has a housing loan (binary: yes, no).",
        "loan": "Whether the client has a personal loan (binary: yes, no).",
        "contact": "Contact communication type for the last contact "
                   "(categorical: unknown, telephone, cellular).",
        "day": "Day of the month of the last contact (numeric).",
        "month": "Month of the year of the last contact (categorical: jan "
                 "through dec).",
        "duration": "Duration of the last contact in seconds (numeric).",
        "campaign": "Number of contacts performed during this campaign for "
                    "this client, including the last contact (numeric).",
        "pdays": "Number of days since the client was last contacted in a "
                 "previous campaign (numeric). A value of -1 means the client "
                 "was not previously contacted.",
        "previous": "Number of contacts performed before this campaign for "
                    "this client (numeric).",
        "poutcome": "Outcome of the previous marketing campaign (categorical: "
                    "unknown, other, failure, success).",
        "y": "Whether the client subscribed to a term deposit (binary: yes, no). "
             "This is the campaign outcome.",
    },

    # ---------------------------------------------------------------- C2
    "diabetes_130us": {
        "encounter_id": "Unique identifier for the hospital encounter.",
        "patient_nbr": "Unique identifier for the patient. A patient may appear "
                       "in multiple encounters.",
        "race": "Patient race (Caucasian, AfricanAmerican, Hispanic, Asian, "
                "Other). Missing values are recorded as '?'.",
        "gender": "Patient gender (Male, Female, or Unknown/Invalid).",
        "age": "Age grouped into ten-year intervals, recorded as a bracketed "
               "string such as [0-10) through [90-100).",
        "weight": "Patient weight in pounds, recorded as a bracketed range. "
                  "Missing values are recorded as '?'; this field is "
                  "unrecorded for the large majority of encounters.",
        "admission_type_id": "Integer code for the admission type, such as "
                             "emergency, urgent, elective, newborn, or not "
                             "available.",
        "discharge_disposition_id": "Integer code for the discharge "
                                    "disposition, such as discharged to home, "
                                    "transferred, or expired.",
        "admission_source_id": "Integer code for the admission source, such as "
                               "physician referral, emergency room, or "
                               "transfer from another facility.",
        "time_in_hospital": "Number of days between admission and discharge "
                            "(integer).",
        "payer_code": "Code for the payer or insurer. Missing values are "
                      "recorded as '?'.",
        "medical_specialty": "Specialty of the admitting physician. Missing "
                             "values are recorded as '?'.",
        "num_lab_procedures": "Number of laboratory tests performed during the "
                              "encounter.",
        "num_procedures": "Number of procedures other than laboratory tests "
                          "performed during the encounter.",
        "num_medications": "Number of distinct generic drug names administered "
                           "during the encounter.",
        "number_outpatient": "Number of outpatient visits by the patient in the "
                             "year preceding the encounter.",
        "number_emergency": "Number of emergency visits by the patient in the "
                            "year preceding the encounter.",
        "number_inpatient": "Number of inpatient visits by the patient in the "
                            "year preceding the encounter.",
        "diag_1": "Primary diagnosis, coded as an ICD-9 code. Missing values "
                  "are recorded as '?'.",
        "diag_2": "Secondary diagnosis, coded as an ICD-9 code. Missing values "
                  "are recorded as '?'.",
        "diag_3": "Additional secondary diagnosis, coded as an ICD-9 code. "
                  "Missing values are recorded as '?'.",
        "number_diagnoses": "Number of diagnoses entered into the system for "
                            "this encounter.",
        "max_glu_serum": "Glucose serum test result. Values are '>200', "
                         "'>300', 'Norm' if within normal range, and 'None' if "
                         "the test was not measured during the encounter.",
        "A1Cresult": "Haemoglobin A1c test result. Values are '>8', '>7', "
                     "'Norm' if within normal range, and 'None' if the test "
                     "was not measured during the encounter.",
        "change": "Whether there was a change in diabetic medications, either "
                  "dosage or generic name ('Ch' for changed, 'No' otherwise).",
        "diabetesMed": "Whether any diabetic medication was prescribed during "
                       "the encounter ('Yes' or 'No').",
        "readmitted": "Readmission outcome: '<30' if the patient was readmitted "
                      "within 30 days, '>30' if readmitted after more than 30 "
                      "days, and 'NO' for no record of readmission.",
        **{col: _MEDICATION_DOC for col in _MEDICATION_COLUMNS},
    },

    # ---------------------------------------------------------------- C3
    "online_retail_ii": {
        "Invoice": "Invoice number, a six-digit nominal code. A code beginning "
                   "with the letter 'c' indicates a cancelled transaction.",
        "StockCode": "Product code, a five-digit nominal code uniquely "
                     "identifying each distinct product.",
        "Description": "Product name.",
        "Quantity": "Quantity of the product in this transaction line. Negative "
                    "quantities occur on returns and cancellations.",
        "InvoiceDate": "Date and time the transaction was generated.",
        "Price": "Unit price of the product in pounds sterling.",
        "Customer ID": "Five-digit nominal code uniquely identifying each "
                       "customer. Blank where the transaction was not "
                       "associated with a registered customer.",
        "Country": "Name of the country where the customer resides.",
    },

    # ---------------------------------------------------------------- C4
    "nyc_tlc_yellow": {
        "VendorID": "Code indicating the technology provider that supplied the "
                    "record. Provider codes have been added over time.",
        "tpep_pickup_datetime": "Date and time the taximeter was engaged.",
        "tpep_dropoff_datetime": "Date and time the taximeter was disengaged.",
        "passenger_count": "Number of passengers in the vehicle. This value is "
                           "entered by the driver.",
        "trip_distance": "Elapsed trip distance in miles as reported by the "
                         "taximeter.",
        "RatecodeID": "Final rate code in effect at the end of the trip: "
                      "1 standard rate, 2 JFK, 3 Newark, 4 Nassau or "
                      "Westchester, 5 negotiated fare, 6 group ride, and "
                      "99 for null or unknown.",
        "store_and_fwd_flag": "Whether the record was held in vehicle memory "
                              "before being sent to the vendor because the "
                              "vehicle had no server connection ('Y' for store "
                              "and forward, 'N' otherwise).",
        "PULocationID": "TLC taxi zone in which the taximeter was engaged.",
        "DOLocationID": "TLC taxi zone in which the taximeter was disengaged.",
        "payment_type": "Numeric code for how the passenger paid: 0 flex fare "
                        "trip, 1 credit card, 2 cash, 3 no charge, 4 dispute, "
                        "5 unknown, and 6 voided trip.",
        "fare_amount": "Time-and-distance fare calculated by the meter.",
        "extra": "Miscellaneous extras and surcharges, such as rush hour and "
                 "overnight charges.",
        "mta_tax": "MTA tax triggered automatically based on the metered rate "
                   "in use.",
        "tip_amount": "Tip amount. This field is populated automatically for "
                      "credit card tips. Cash tips are not recorded.",
        "tolls_amount": "Total amount of all tolls paid during the trip.",
        "improvement_surcharge": "Improvement surcharge assessed at the flag "
                                 "drop on hailed trips.",
        "total_amount": "Total amount charged to the passenger. Does not "
                        "include cash tips.",
        "congestion_surcharge": "Total amount collected for the New York State "
                                "congestion surcharge.",
        "Airport_fee": "Fee applied to pickups at LaGuardia and John F. "
                       "Kennedy airports.",
        "cbd_congestion_fee": "Per-trip charge for the MTA Congestion Relief "
                              "Zone, applicable to trips from 2025 onward.",
    },
}


def get_docs(corpus_id: str) -> dict[str, str]:
    """Column documentation for one corpus. Empty dict if none is registered."""
    return dict(DATA_DICTIONARIES.get(corpus_id, {}))


def get_table_doc(corpus_id: str) -> str:
    """Table-level description for one corpus."""
    return TABLE_DESCRIPTIONS.get(corpus_id, "")


def coverage_report(corpus_id: str, columns: list[str]) -> dict[str, object]:
    """Check documentation coverage against a corpus's actual columns.

    A3 is only a fair condition if the documentation covers the schema. An
    undocumented column silently makes A3 partially equivalent to A2 for that
    column, so run this before generating and act on anything it reports.
    """
    docs = get_docs(corpus_id)
    documented = [c for c in columns if c in docs]
    undocumented = [c for c in columns if c not in docs]
    orphaned = [c for c in docs if c not in columns]
    return {
        "corpus_id": corpus_id,
        "n_columns": len(columns),
        "n_documented": len(documented),
        "coverage": round(len(documented) / len(columns), 4) if columns else 0.0,
        "undocumented": undocumented,
        "orphaned_docs": orphaned,
    }
