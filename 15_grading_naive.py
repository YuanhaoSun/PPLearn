import re

f = open('16.txt', 'rb')

text = f.read()

# regex matches
cookies = re.search('cooki', text, re.IGNORECASE)
ssl = re.search('(\WSSL|encrypt|safeguard)', text, re.IGNORECASE)
children = re.search('children', text, re.IGNORECASE)
safeharbor = re.search('safe.?harbor', text, re.IGNORECASE)
truste = re.search('truste', text, re.IGNORECASE)
third = re.search('third.?part', text, re.IGNORECASE)
choice = re.search('(choice|opt.?(out|in))', text, re.IGNORECASE)
ads = re.search('(\sads|advertis)', text, re.IGNORECASE) #Only matech ads with a space in front
collect = re.search('collect', text, re.IGNORECASE)
share = re.search('(share|disclos)', text, re.IGNORECASE)

# need to count location, so use re.findall
location = re.findall('location', text, re.IGNORECASE)


grade = 0.0

if cookies:
	grade += 1
if ssl:
	grade += 1
if children:
	grade += 1
if safeharbor:
	grade += 2
if truste:
	grade += 1
if third:
	grade += 1
if choice:
	grade += 1
if ads:
	grade += 1
if collect:
	grade += 1
if share:
	grade += 1
if len(location) >= 5:
	grade += 0.5

if grade > 10:
	grade = 10

print grade