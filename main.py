try:
    import en_core_web_sm
except ModuleNotFoundError:
    import subprocess
    subprocess.call('pip install --upgrade --user spacy')
    subprocess.call('python -m spacy en_core_web_sm')

import en_core_web_sm, random
from os.path import dirname, abspath
from time import perf_counter
from pandas import read_excel
from spacy.util import minibatch, compounding
from spacy import displacy
current_location = dirname(abspath(__file__))

s = perf_counter()

df = read_excel('./data/indeed_results.xlsx')

description = df['Description'].dropna().reset_index()['Description']
pre_processed=description.apply(lambda x: str(x).replace('Job Summary','').replace('Job Description','').replace('Short Description','').replace('\n',' ').replace('(',' ').replace(')',' ').replace('/',' ').replace('|',' ').replace('  ',' ').replace('"',' ').replace("'",' ').replace('  ','').lstrip().rstrip())

nlp = en_core_web_sm.load()
ner = nlp.get_pipe('ner')

# JT -> Job Timing
# ORG -> Organization
# JR -> Job Role
# JS -> Job Skill
# QN -> Qualification
# DOC -> Documents
# PL -> Programming Language
# PF -> Programming Framework
train = [
    (
        'Official Job posted from SwiggyJoin Swiggy family as Part-Time or a Full-time Food Delivery executive.', 
        {'entities':[(25,31,'ORG'),(36,42,'ORG'),(53,63,'JT'),(68,77,'JT'),(78, 101, 'JR')]}
    ),
    (
        'Who can be an ideal Rider Driver Delivery Boy Logistics AdminDelivery Executive for Swiggy?', 
        {'entities':[(14,32,'JS'),(33,45,'JS'),(46,61,'JS'),(61,79,'JS'),(84,90,'ORG')]}
    ),
    (
        'Minimum qualification - Fresher, 10th pass, 12th pass, Graduate, New Graduates, Trainee Candidates',
        {'entities':[(24,31,'QN'),(33,42,'QN'),(42,53,'QN'),(55,63,'QN'),(65,78,'QN'),(80,99,'QN')]}
    ),
    (
        'with prior experience in Data entry, Call Centre, Admin, Collection Agents, Collection Executives, Customer Support Executive, Office Assistant, Driver, Delivery Boys, Back Office, Teacher, Banking, Accounts, Operator can apply to maximize their earnings up to 25,000 per month.',
        {'entities':[(25,35,'JR'),(37,48,'JR'),(50,55,'JR'),(57,74,'JR'),(76,96,'JR'),(98,125,'JR'),(127,143,'JR'),(145,151,'JR'),(153,166,'JR'),(168,179,'JR'),(181,188,'JR'),(190,197,'JR'),(199,207,'JR'),(209,217,'JR')]}
    ),
    (
        'Documents to be carried for Walk-in: Valid Two-wheeler Driver License and Driver RC book PAN Card or',
        {'entities':[(37,69,'DOC'),(74,88,'DOC'),(89,97,'DOC')]}
    ),
    (
        'Acknowledgment of PAN application Aadhaar Card or voter ID Benefits:',
        {'entities':[(0,33,'DOC'),(34,46,'DOC'),(49,58,'DOC')]}
    ),
    (
        'Flexible working hours for security guards, computer operators,',
        {'entities':[(27,42,'JR'),(44,62,'JR')]}
    ),
    (
        'account executives night shifts weekend Attractive Weekly, weekend and monthly Incentives for 10th pass, 12th pass',
        {'entities':[(0,18,'JR'),(19,31,'JT'),(32,39,'JT'),(94,103,'QN'),(105,114,'QN')]}
    )
]

for _, annotations in train:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])
        
disable_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

print('Before Training')
for text, _ in train:
    doc = nlp(text)
    print(doc)
    print(f'Entities: {[(ent.text, ent.label_) for ent in doc.ents]}')

doc = nlp('Official Job posted from SwiggyJoin Swiggy family as Part-Time or a Full-time Food Delivery executive. Who can be an ideal Rider Driver Delivery Boy Logistics AdminDelivery Executive for Swiggy? Minimum qualification - Fresher, 10th pass, 12th pass, Graduate, New Graduates, Trainee Candidates with prior experience in Data entry, Call Centre, Admin, Collection Agents, Collection Executives, Customer Support Executive, Office Assistant, Driver, Delivery Boys, Back Office, Teacher, Banking, Accounts, Operator can apply to maximize their earnings up to 25,000 per month. Documents to be carried for Walk-in: Valid Two-wheeler Driver License and Driver RC book PAN Card or Acknowledgment of PAN application Aadhaar Card or voter ID Benefits: Flexible working hours for security guards, computer operators, account executives night shifts weekend Attractive Weekly, weekend and monthly Incentives for 10th pass, 12th pass')
for ent in doc.ents:
    print(f'Text: {ent.text:4}\t\t Start Char: {ent.start_char:4}\t\t End Char: {ent.end_char}\t\t Label: {ent.label_}')

print('Training')
with nlp.disable_pipes(*disable_pipes):
    optimizer = nlp.resume_training()
    
    for iteration in range(100):
        random.shuffle(train)
        losses = {}
        
        batches = minibatch(train, size=compounding(1.0, 4.0, 1.001))
        
        for batch in batches:
            text, annotation = zip(*batch)
            nlp.update(text, annotation, drop=0.5, losses=losses, sgd=optimizer)

            print(f'Losses: {losses}')
            
print('After Training')
for text, _ in train:
    doc = nlp(text)
    print(doc)
    print(f'Entities: {[(ent.text, ent.label_) for ent in doc.ents]}')

doc = nlp('Official Job posted from SwiggyJoin Swiggy family as Part-Time or a Full-time Food Delivery executive. Who can be an ideal Rider Driver Delivery Boy Logistics AdminDelivery Executive for Swiggy? Minimum qualification - Fresher, 10th pass, 12th pass, Graduate, New Graduates, Trainee Candidates with prior experience in Data entry, Call Centre, Admin, Collection Agents, Collection Executives, Customer Support Executive, Office Assistant, Driver, Delivery Boys, Back Office, Teacher, Banking, Accounts, Operator can apply to maximize their earnings up to 25,000 per month. Documents to be carried for Walk-in: Valid Two-wheeler Driver License and Driver RC book PAN Card or Acknowledgment of PAN application Aadhaar Card or voter ID Benefits: Flexible working hours for security guards, computer operators, account executives night shifts weekend Attractive Weekly, weekend and monthly Incentives for 10th pass, 12th pass')
for ent in doc.ents:
    print(f'Text: {ent.text:4}\t\t Start Char: {ent.start_char:4}\t\t End Char: {ent.end_char}\t\t Label: {ent.label_}')

e = perf_counter()
print(f'Time Taken: {e-s:.2f} seconds')