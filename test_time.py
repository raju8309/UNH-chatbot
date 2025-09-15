import time
# from main import load_json_file, answer_question
from main import load_json_file, cached_answer_str as answer_question

# load data before testing
load_json_file("course_descriptions.json")
load_json_file("degree_requirements.json")

# warm up model
_ = answer_question("Warmup query")

# list of test questions
questions = [
    "Does UNH offer any graduate courses in health-related fields?",
    "What courses are listed under Accounting and Finance (ACFI)?",
    "Show me the Computer Science (CS) course descriptions.",
    "Give me the link to Zoology (ZOOL) courses.",
    "Are there any graduate courses in Public Health or Healthcare?",
    "What are the graduate courses offered in music?",
    "What courses fall under Biology-related subjects?",
    "Show me everything available under Education (EDUC/EDC).",
    "What are the graduate degree requirements at UNH?",
    "What is the time limit for completing a graduate degree?",
    "What are the requirements for earning a Ph.D. at UNH?",
    "Is the Ph.D. at UNH a research-based degree?",
    "Does UNH offer an OTD (Doctor of Occupational Therapy) program?",
    "What is Responsible Conduct of Research (RCR) training, and is it required for Ph.D. students?",
    "When should a Ph.D. student form a guidance committee?",
    "What are the residency requirements for a Ph.D.?",
    "Do Ph.D. students need to register for Doctoral Research (999) every semester?",
    "How many years of graduate study are required for the Ph.D.?",
    "When is a Ph.D. student advanced to candidacy?",
    "Are there language or research proficiency requirements for doctoral students?",
    "How many members are on a doctoral committee?",
    "What are the requirements for defending and submitting a dissertation?",
    "Does the dissertation need to be submitted through ProQuest?",
    "How long do I have to complete my Ph.D.?",
    "What is the time limit for completing the Ed.D., DNP, or OTD?",
    "If I start a Ph.D. with a master’s degree, does the time limit change?",
    "What is the minimum number of credits required for a master’s degree?",
    "Can 700-level courses count toward a master’s degree?",
    "Does every master’s program require a capstone experience?",
    "How long do students have to finish a master’s degree?",
    "Can Ph.D. students petition for a non-terminal master’s degree along the way?",
    "Who serves on a master’s thesis committee?",
    "Where do I submit my final thesis?",
    "What are the requirements for the Educational Specialist degree?",
    "Can prior credits be applied toward the Ed.S.?",
    "How long do I have to complete an Ed.S. degree?",
    "How many credits are required for a graduate certificate?",
    "What is the time limit for completing a certificate program?",
    "Can certificate courses also count toward a master’s or Ph.D.?",
    "Are certificate students eligible for financial aid or assistantships?",
    "What is the tuition policy for certificate programs?",
    "Can I pursue two graduate degrees at the same time at UNH?",
    "What are the different types of dual degrees offered?"
]

# get response times
times = []

for q in questions:
    start = time.time()
    answer = answer_question(q)
    end = time.time()
    times.append(end - start)
    print(f"Question: {q}")
    print(f"Response time: {end - start:.3f} sec\n")

avg_time = sum(times) / len(times)
print(f"average response time: {avg_time:.3f} sec")
