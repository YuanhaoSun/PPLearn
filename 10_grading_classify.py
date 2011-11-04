"""
======================================================
Grading - base on classification resuts
======================================================
Give grade on a url link or html file based on 
the coverage of topics using classification
"""

from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction.text import Vectorizer

from sklearn.datasets import load_files
from sklearn.externals import joblib

from utils.html2txt import Html2txtConverter 

####################################################################
# Cutomized grading scheme
# 
# Default grading scheme:
# assume: E(topic)=1 if topic exists, or E(topic)=0 if it doesn not exist
# Grade = ( E(ads)+E(ca)+E(collect)+E(cookies)+E(security)+E(share)
#           +1.5*E(safeharbor)
#           +0.5*(E(change)+E(children)+E(contact)+E(location)+E(process)+E(retention)+E(truste))
#          )
# Grade = 10 if Grade > 10
# 

def grading(predicted_cate_names):
    
    # predefine the list of categories in three priorities for further grading
    high = ['SafeHarbor']
    medium = set(('Advertising', 'CA', 'Collect', 'Cookies', 'Security', 'Share'))
    low = set(('Change', 'Children', 'Contact', 'Location', 'Process', 'Retention', 'Truste'))

    grade = 0
    
    for item in predicted_cate_names:
        if item in high:
            grade += 1.5
        elif item in medium:
            grade += 1.0
        elif item in low:
            grade += 0.5
        else:
            print 'Error - found an unmatched name during grading:'
            print ('%s!')% item
    
    # deal with situation when grade > 10
    if grade > 10:
        grade = 10

    return grade


####################################################################
# Cutomized grading function using classification results
# gets a predicted_cate and cate_names, returns the grade
#
# parameters:
# predicted_cate: the predicted results on a given test item, List of numbers
# cate_names: standard categories from the train data
#

def grade_privacypolicy(predicted_cate, standard_cate_names):

    # remove duplications in predicted_cate list
    predicted_no_dup = list(set(predicted_cate))

    # get correct category names for predicted
    # using the standard_cate_names
    predicted_cate_names = []
    for item in predicted_no_dup:
        predicted_cate_names.append(standard_cate_names[item])

    # call the grading function to grade on predicted_cate_names
    grade = grading(predicted_cate_names)

    return (grade, predicted_cate_names)


####################################################################
# Cutomized filtering function to return the score for the decision
# given a decision_function and multi_label_threshold
# 
# parameters:
# multi_label_threshold: threshold to feedback 0-n labels, whatever bigger than threshold
# defaults is set to 111 as dummy value. Because False cannot be applied due to situation
# where False(==0) can be used as an actual threshold
# 

def filter_decision_multi(decision_function, multi_label_threshold=111):

    # Multi-label threshold
    if multi_label_threshold != 111:

        filtered_decision_function = []

        for decision_item in decision_function:

            # transfer decision_function to list of tuples -- (score, #)
            # '#' is for future reference of label
            decision_item_num = []
            for i in range(len(decision_item)):
                decision_item_num.append((decision_item[i], i))

            # sort base on score. # will not lost
            decision_item_sorted =  sorted(decision_item_num)
            decision_item_sorted.reverse()

            # filter by multi_label_threshold
            # using paradigm: li = [ x for x in li if condition(x)] to
            # filter base on a condition(x)
            decision_item_filtered = [ x for x in decision_item_sorted if x[0]>multi_label_threshold ]

            # append this item back to the filtered decision function
            filtered_decision_function.append(decision_item_filtered)

        # filtered_decision_function here is a list of list of tuple, 
        # e.g. in case of 3 paragraphs:
        # [
        #   [(score of first label,#), (score of second label,#), (score of third label,#)]
        #   [(score of first label,#), (score of second label,#)]
        #   [(score of first label,#), (score of second label,#), (score of third label,#)]
        # ]
        return filtered_decision_function


####################################################################
# Cutomized filtering function to return the score for the decision
# given a decision_function and one_class_threshold
# 
# parameters:
# one_class_threshold: threshold to filter the biggest label
# default is set to 111 as dummy value. Because False cannot be applied due to situation
# where False(==0) can be used as an actual threshold
# 

def filter_decision_one(decision_function, one_label_threshold=111):

    # One-label threshold
    if one_label_threshold != 111:

        filtered_decision_function = []

        for decision_item in decision_function:

            # transfer decision_function to list of tuples -- (score, #)
            # '#' is for future reference of label
            decision_item_num = []
            for i in range(len(decision_item)):
                decision_item_num.append((decision_item[i], i))

            # sort base on score. # will not lost
            decision_item_sorted =  sorted(decision_item_num)
            decision_item_sorted.reverse()

            # get the top tuple with highest score
            decision_item_top_tuple = decision_item_sorted[0]

            # if the score of the top tuple is bigger than threshold, add to list
            if decision_item_top_tuple[0] >= one_label_threshold:
                filtered_decision_function.append(decision_item_top_tuple)
            # else, it is smaller, add a sign the means no label is marked    
            else:
                filtered_decision_function.append((0, 99)) #99 will be a signal for no label

        # filtered_decision_function here is a list of tuple, 
        # e.g. in case of 3 paragraphs, if their top labels all bigger than the threshold
        # [(score of top label,#), (score of top label,#), (score of top label,#)]
        return filtered_decision_function





# categories = ['Advertising','CA', 'Collect', 'Cookies']
categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
            'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
            'Process', 'Retention']


# Load all training datasets
print "Loading privacy policy datasets..."
data_train = load_files('Privacypolicy/raw', categories = categories,
                        shuffle = True, random_state = 42)
print 'Data loaded!'
print

standard_cate_names = data_train.target_names

# load models from pickle files
clf = joblib.load('models/classifier_Ridge.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')


# Testset - X_new
# docs_new = ["Eli Lilly and Company complies with the U.S.-EU Safe Harbor Framework and the U.S.-Swiss Safe Harbor Framework as set forth by the U.S. Department of Commerce regarding the collection, use, and retention of personal information from European Union member countries and Switzerland. Eli Lilly and Company has certified that it adheres to the Safe Harbor Privacy Principles of notice, choice, onward transfer, security, data integrity, access, and enforcement. To learn more about the Safe Harbor program, and to view Eli Lilly and Company's certification, please visit", 
#             'Through this website, you may be asked to voluntarily provide to us information that can identify or locate you, such as your name, address, telephone number, e-mail address, and other similar information ("Personal Information"). You may always refuse to provide Personal Information to us, but this may lead to our inability to provide you with certain information, products or services.',
#             'We may also collect information that does not identify you ("Other Information"), such as the date and time of website session, movements across our website, and information about your browser, We do not tie Other Information to Personal Information. This Other Information may be used and stored by Lilly or its agents. Other Information we collect may include your IP Address (a number used to identify your computer on the Internet) and other information gathered through our weblogs, cookies or by other means (see below). We use Other Information to administer and update our website and for other everyday business purposes. Lilly reserves the right to maintain, update, disclose or otherwise use Other Information, without limitation.',
#             'This website may use a technology called a "cookie". A cookie is a piece of information that our webserver sends to your computer (actually to your browser file) when you access a website. Then when you come back our site will detect whether you have one of our cookies on your computer. Our cookies help provide additional functionality to the site and help us analyze site usage more accurately. For instance, our site may set a cookie on your browser that keeps you from needing to remember and then enter a password more than once during a visit to the site. With most Internet browsers or other software, you can erase cookies from your computer hard drive, block all cookies or receive a warning before a cookie is stored. Please refer to your browser instructions to learn more about these functions. If you reject cookies, functionality of the site may be limited, and you may not be able to take advantage of many of the site features',
#             'This website collects and uses Internet Protocol (IP) Addresses. An IP Address is a number assigned to your computer by your Internet service provider so you can access the Internet. Generally, an IP address changes each time you connect to the Internet (it is a "dynamic" address). Note, however, that if you have a broadband connection, depending on your individual circumstance, it is possible that your IP Address that we collect, or even perhaps a cookie we use, may contain information that could be deemed identifiable. This is because with some broadband connections your IP Address does not change (it is "static") and could be associated with your personal computer. We use your IP address to report aggregate information on use and to help improve the website.',
#             'Except as provided below, Lilly will not share your Personal Information with third parties unless you have consented to the disclosure',
#             'We may share your Personal Information with our agents, contractors or partners in connection with services that these individuals or entities perform for, or with, Lilly, such as sending email messages, managing data, hosting our databases, providing data processing services or providing customer service. These agents, contractors or partners are restricted from using this data in any way other than to provide services for Lilly, or services for the collaboration in which they and Lilly are engaged (for example, some of our products are developed and marketed through joint agreements with other companies)',
#             'Lilly will share Personal Information to respond to duly authorized subpoenas or information requests of governmental authorities, to provide Internet security or where required by law. In exceptionally rare circumstances where national, state or company security is at issue, Lilly reserves the right to share our entire database of visitors and customers with appropriate governmental authorities',
#             'We may also provide Personal Information and Other Information to a third party in connection with the sale, assignment, or other transfer of the business of this website to which the information relates, in which case we will require any such buyer to agree to treat Personal Information in accordance with this Privacy Statement',
#             'Lilly will respect your marketing and communications preferences. You can opt-out of receiving commercial emails from us at any time by following the opt-out instructions in our commercial emails. You can also request removal from all our contact lists, by writing to us at the following address',
#             'Lilly encourages parents and guardians to be aware of and participate in their children online activities. You should be aware that this site is not intended for, or designed to attract, individuals under the age of 18. We do not collect Personal Information from any person we actually know to be under the age of 18',
#             'Lilly protects all Personal Information with reasonable administrative, technical and physical safeguards. For example, areas of this website that collect sensitive Personal Information use industry standard secure socket layer encryption (SSL); however, to take advantage of this your browser must support encryption protection (found in Internet Explorer release 3.0 and above).',
#             'As a convenience to our visitors, this Website currently contains links to a number of sites owned and operated by third parties that we believe may offer useful information. The policies and procedures we describe here do not apply to those sites. Lilly is not responsible for the collection or use of Personal Information or Other Information at any third party sites. Therefore, Lilly disclaims any liability for any third party use of Personal Information or Other Information obtained through using the third party web site. We suggest contacting those sites directly for information on their privacy, security, data collection, and distribution policies.',
#             'We may update this Privacy Statement from time to time. When we do update it, for your convenience, we will make the updated statement available on this page. We will always handle your Personal Information in accordance with the Privacy Statement in effect at the time it was collected. We will not make any materially different use or new disclosure of your Personal Information unless we notify you and give you an opportunity to object',
#             'If you have any questions or comments about this Privacy Statement, please contact us by writing to:',
#             "Greece's leaders scrambled to restore political stability in the country and preserve it's euro membership, killing a controversial plan for a referendum on Greece's latest bailout that has roiled global markets.",
#             "After being flooded with calls, faxes and emails calling for action, a Texas judicial panel is investigating an internet video that shows a judge beating his teenage daughter with a belt.",
#             "Apple has acknowledged some customers drain their batteries unusually quickly when the use its new iOS 5 operating system on the iPhone 4S or other devices.",
#             "A rift has formed in the shelf of floating ice in front of the Pine Island Glacier (PIG). The crack in the PIG runs for almost 30km (20 miles), is 60m (200ft) deep and is growing every day.",
#             "Saber-toothed squirrel: With its superlong fangs, long snout and large eyes, the mouse-size animal bears an oddly striking resemblance to the fictional saber-toothed squirrels depicted in the computer-animated 'Ice Age' films, scientists added.",
#             "Occupy Wall Street supporters who staged rallies that shut down the nation's fifth-busiest port during a day of protests condemned on Thursday the demonstrators who clashed with police in the latest flare-up of violence in Oakland.",
#             "An Afghan police officer walks at the scene of a suicide attack in Herat, west of Kabul, Afghanistan. A suicide car bomb struck the compound of a NATO contractor company in western Afghanistan while other heavily-armed insurgents",
#             "Two stoners will probably have trouble stealing the loot from a group of thieves at the box office. Tower Heist, a comedy starring Eddie Murphy and Ben Stiller about a bunch of crooks attempting to pull off a robbery, is poised to run away with the",
#             ]
# docs_new = [
# "This website is for your general information and use only. It is subject to change without notice. All availability times and images are provided as a guide, we cannot be held liable if it takes a little longer to make or if it differs slightly from the image as all items are handmade, and so will not look exact each time.",
# "We reserve the right to change the prices of items at any time. All prices on the website are in UK Pounds Sterling.",
# "Please read the following explanation for the information collected and how it is used, how it is safeguarded and how to contact us if you have any concerns regarding our Privacy Policy.",
# "The following information is collected from shoppers as part of the order process: Personal name, Shipping/Billing Address, Email Address.",
# "Your name and shipping address are only used in order to ship your items to you.",
# "Email addresses are used in order to contact you regarding an order you have placed with us. This may involve the following: confirmation of your order, informing you of shipment and/or any additional information about your order. ",
# "You will not be contacted for reason not relating to your order.",
# "You're email address will only be used for the mailing list if you have chosen to subscribe either via the mailing list facilty on the webiste or in person. If you ever wish to unsubscribe from the mailing list please use the 'unsubscribe' link provided in every email or contact us. You're email address will never been forwarded to third parties.",
# "Personal names, shipping addresses, and email addresses are never used for other purposes or shared with anyone else.",
# "All transactions are done through Paypal which keeps your credit/debit card information private. Thus, credit/debit card numbers are never made known to us.",
# "Please do not hesitate to contact us should you have any queries regarding our Privacy Policy.",
# ]

# instead of using a fixed list of paragraphs, we now use utils.html2txt
# to open url and get back a list of paragraphs
# link = 'https://cms.paypal.com/us/cgi-bin/?cmd=_render-content&content_ID=ua/Privacy_print'
link = 'http://www.lilly.com/privacy/Pages/default.aspx'
html_converter = Html2txtConverter(url=link, length_threshold=150)
docs_new = html_converter.get_paragraphs()

X_new = vectorizer.transform(docs_new)

####################################################
# Predict

# get decision_function
print clf
pred_decision = clf.decision_function(X_new)
# print pred_decision
print

# # predict by simply apply the classifier
# # this will not use any threshold (neither one-label nor multi-label thresholds)
# predicted = clf.predict(X_new)
# for doc, category in zip(docs_new, predicted):
#     print '%r => %s' % (doc, standard_cate_names[int(category)])
#     print

# predict with one_label_threshold
# The highest score label in each prediction for one paragraph will
# be stored only if it is bigger than the threshold
predicted = filter_decision_one(pred_decision, one_label_threshold=0.25)
# print predicted
# print

# print the filtered predict result
for doc, label in zip(docs_new, predicted):
    print doc
    # deal with the signal '99' aka 'no label' aka 'this paragraph is not privacy policy'
    if label[1] == 99:
        print 'Not a privacy policy'
    else:
        print data_train.target_names[label[1]], label[0]
    print


####################################################
# Grading

# get the number-only list of categories for grading
predicted_cate = []
for item in predicted:
    # only add # (i.e. item[1]) into the list
    # also remove the signal '99' for 'no label'
    if item[1] != 99:
        predicted_cate.append(item[1])
print predicted_cate

# get grading result -- (grade, [covered topics])
result = grade_privacypolicy(predicted_cate, standard_cate_names)

print 'Grade: %.1f' % result[0]
print 'Covers:'
print result[1]