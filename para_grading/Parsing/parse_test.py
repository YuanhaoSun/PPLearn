import re

ex = r'([\w-]*)\(([\w-]*)-\d*, ([\w-]*)-\d*\)'

lines = [ "dep(Yahoo-1, rent-5)",  "conj_or(rent-5, rent-5')", "conj_or(rent-5, rent-5')"]

for line in lines:
	# m = re.search('(?<=())\w+', line)
	# m2 =re.search('(?<=-)\d', line)
	# print m.group(0)
	# print m2.group(0)
	m = re.match(ex, line)
	print m.group(0)
	# print(m.group(0), m.group(1), m.group(2))

