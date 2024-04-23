import ast
import os
import pickle
import re
import time
import timeit
import warnings
from datetime import datetime
from pathlib import Path

import Levenshtein
import pandas as pd
import pytz
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from tqdm import TqdmWarning

import constants as const

# Suppress specific warnings
warnings.filterwarnings('ignore', category=TqdmWarning)

warnings.filterwarnings('ignore')
ist_timezone = pytz.timezone('Asia/Kolkata')

load_dotenv()


def bucket_budget(budget):
    budget = int(budget)
    if budget < 0:
        return 'Invalid Budget'

    budget_buckets = {
        '0_to_50': 'Mercury_Swift',
        '50_to_100': 'Venus_Bright',
        '100_to_150': 'Earth_Blue',
        '150_to_200': 'Mars_Red',
        '200_to_250': 'Jupiter_Giant',
        '250_to_300': 'Saturn_Ringed',
        '300_to_350': 'Uranus_Icy',
        '350_to_400': 'Neptune_Distant',
        '400_to_450': 'Pluto_Dwarf',
        '450_to_500': 'Ceres_Asteroid',
        '500_to_550': 'Pallas_Belted',
        '550_to_600': 'Vesta_Cratered',
        '600_to_650': 'Hygiea_Rocky',
        '650_to_700': 'Eris_Discovered',
        '700_to_750': 'Haumea_Elliptical',
        '750_to_800': 'Makemake_Kuiper',
        '800_to_850': 'Sedna_Far',
        '850_to_900': 'Orcus_Vault',
        '900_to_950': 'Quaoar_Resonant',
        '950_to_1000': 'Gonggong_Water',
        '1000_to_1050': 'Andromeda_Galaxy',
        '1050_to_1100': 'Orion_Hunter',
        '1100_to_1150': 'Pegasus_Winged',
        '1150_to_1200': 'Cassiopeia_Queen',
        '1200_to_1250': 'Centaurus_Half',
        '1250_to_1300': 'Draco_Dragon',
        '1300_to_1350': 'Gemini_Twins',
        '1350_to_1400': 'Hercules_Mighty',
        '1400_to_1450': 'Lyra_Harp',
        '1450_to_1500': 'Perseus_Hero',
        '1500_to_1550': 'Phoenix_Fire',
        '1550_to_1600': 'Sagittarius_Archer',
        '1600_to_1650': 'Scorpius_Scorpion',
        '1650_to_1700': 'Taurus_Bull',
        '1700_to_1750': 'UrsaMajor_Bear',
        '1750_to_1800': 'UrsaMinor_Little',
        '1800_to_1850': 'Virgo_Maiden',
        '1850_to_1900': 'Aquarius_WaterBearer',
        '1900_to_1950': 'Aries_Ram',
        '1950_to_2000': 'Leo_Lion',
    }

    # Define a list of English alphabet words for numbers up to 2000.
    words = [str(i) for i in range(0, 2001, 50)]

    if budget > 2000:
        return 'Budget_Excess_Opulent_Option'

    bucket_label = words[int(budget / 50)]
    bucket_label += '_to_' + words[int(budget / 50) + 1]

    return 'Budget_tag: ' + budget_buckets[bucket_label]


def p_name(paragraph):
    # Use regular expression to find the property name
    match = re.search(r'Property: ([^,]+),', paragraph)

    # Check if a match is found
    if match:
        property_name = match.group(1).strip()
        return property_name
    else:
        return None


# base_folder = '/Users/apple/Documents/Bot/Vectors/cities/'
base_folder = Path.cwd() / 'vectors' / 'cities'

city_folder_names = [folder.name for folder in base_folder.iterdir() if folder.is_dir()]


def property_manager_encoded(text, kind):
    if kind == 2:
        start_phrase = 'property manager:'
        end_phrase = '. QUESTION'
    if kind == 1:
        start_phrase = 'property manager'
        end_phrase = '. Starting price'

    # Find the index of the start and end phrases
    start_idx = text.find(start_phrase) + len(start_phrase)
    end_idx = text.find(end_phrase)

    # Find the index of the first colon after the start phrase
    colon_idx = text.find(':', start_idx)

    # Find the last space before the colon to identify the word to remove
    space_idx = text.rfind(' ', start_idx, colon_idx)

    # Construct the new string by keeping part before the space and part from the end phrase
    new_text = text[:space_idx] + text[end_idx:]

    return new_text


# def contains_negative_words(text):
#     negative_words = ["sorry", "cannot", "can't", "donot", "don't","don't know"]  # Add more negative words as needed
#     pattern = r'\b(?:' + '|'.join(negative_words) + r')\b'
#     match = re.search(pattern, text, flags=re.IGNORECASE)
#     return bool(match)
def correct(text):
    # Check if the first line starts with "Distance"
    if re.match(r'^Distance', text):
        # Remove the first line until the first full stop before ". University"
        text = re.sub(r'^.*?\.', '', text)
        return 'This property is near ' + text[1:]
    return text


negative_words = ['sorry', 'donot', "don't", "don't know"]
pattern = re.compile(r'\b(?:' + '|'.join(negative_words) + r')\b', flags=re.IGNORECASE)


def contains_negative_words(text):
    match = pattern.search(text)
    return bool(match)


chat = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-1106')
# chat = ChatOpenAI(temperature=0,model='gpt-3.5-turbo-0125')


d = {'city': ''}
archive = []
message_history = []


# Example usage:
budget = 0
# bucket = bucket_budget(budget)
# print(f"Budget bucket: {bucket}")
# print(bucket)


def find_closest_match(user_input, folder_names):
    closest_match = min(
        folder_names,
        key=lambda folder: Levenshtein.distance(user_input.lower(), folder.lower()),
    )
    return closest_match


def guarantor_fix(para):
    UK_BASED = 'Agriculture is the science and practice of cultivating crops and raising livestock '
    INTERNATIONAL = (
        'The study of fundamental particles and their behavior at the quantum level'
    )
    NOT_REQUIRED = 'Medical studies involve the comprehensive examination of human health, encompassing diagnosis, treatment'

    if UK_BASED in para:
        para = para.replace(UK_BASED, 'Required UK based local only.')
    elif INTERNATIONAL in para:
        para = para.replace(INTERNATIONAL, 'International accepted.')
    elif NOT_REQUIRED in para:
        para = para.replace(NOT_REQUIRED, 'Not required.')

    return para


def predict(input, status, messages, message_history):
    archive.append(datetime.now())

    if str.lower(input) == 'new':
        df = pd.DataFrame({'conversation': message_history})
        df.to_csv(
            f'{Path.cwd()}/conversations/'
            + str(datetime.now(ist_timezone))
            + '_conversations.csv'
        )
        status = ''

        messages = [
            SystemMessage(
                content="UniAcco specializes in providing student accommodations for students going abroad for studies. Our properties are not for non-students or any one going for trip, we only have rooms for students. On our websites we list various properties only in United Kingdom. I want you to act like an experienced sales executive and talk to our potential customers with the sole motive of completing the sale and matching the demand with the supply. The process is divided into three steps: Step 1: Ask them about their city and university. Step 2: Confirming the step 1 requirement. Step 3: Recommend properties based on the requirements like room types and even the budget range, lease, intake and the amenities they are looking for.\n\n Status of Step 1: In progress. Step 2: In progress. Step 3: Not started. So as of now we will focus on step 1 and 2, and insist student to do the needful. Once student shared his university requirement, then we need to confirm the student's actual university(or may be campus as there are multiple campuses sometimes) and city, this will help map the correct city and university name from our database. So provide the list of closest match universities and ask the student to confirm the university or campus. Once the student confirms city and university requirements, then strictly reply 'CONFIRMED'. You can begin the conversation with welcome message(Hi, welcome to UniAcco-along with this asking about their requirements). Important Note: 1. Do not go astray from the conversation, and converge to a sale. 2. Answer politely that the questions is strictly 'Out of Scope' for irrelevant questions(like who is Vin Diesel or ELon Musk, or book me a flight) or questions that can cause harm to UniAcco public reputation or any sensitive topic. 3. Answer should be in concise and to the point. 4. If during step 1, student's wants room near a university which is not in United Kingdom, or wants property in some other country than UK, strictly say that we currently do not have supply.5. Always try to answer positively about questions related to UniAcco or UniAcoo's website."
            )
        ]

        ai_response = "Hi, welcome to UniAcco! I'd love to help you find the perfect student accommodation. Could you please let me know which city and university you will be studying in?"

        messages.append(AIMessage(content=ai_response))

        message_history = []
        message_history.append({'role': 'user', 'content': f'hey'})
        message_history.append({'role': 'assistant', 'content': f'{ai_response}'})

        # get pairs of msg["content"] from message history, skipping the pre-prompt: here.
        response = [
            (message_history[i]['content'], message_history[i + 1]['content'])
            for i in range(0, len(message_history) - 1, 2)
        ]  # convert to tuples of list

        return response, status, messages, message_history

    '''Find the k best matched chunks to the queried test.
    These will be the context over which our bot will try to answer the question.
    The value of k can be adjusted so as to get the embeddings for the best n chunks.'''

    message_history.append({'role': 'user', 'content': f'{input}'})
    conversation = ''
    for i, message in enumerate(message_history):
        if i > 0:
            if i % 2 == 0:  # Check if the index is even
                conversation += 'Student: ' + message['content'] + '\n\n'
            else:
                conversation += 'Chatbot: ' + message['content'] + '\n\n'

    conversation_system = [
        SystemMessage(
            content="You are tasked with analyzing a conversation between a student and a chatbot from UniAcco, a platform specializing in student accommodations. Your primary objective is to meticulously extract the student's specific accommodation requirements from their latest query. This information should be methodically organized into a Python dictionary named 'd'. The dictionary must include the following keys: 'properties', 'property_manager', 'city', 'university', 'room_type', 'budget', 'lease', 'guarantor', 'dual_occupancy', and 'amenities'. It's crucial to update all keys in accordance with the most recent requirements expressed by the student. For example, if the student initially inquires about rooms in London but later shifts their interest to Manchester near a particular university, the keys should be updated to reflect this latest preference.\n\nThere are two primary cases to consider:\n1. Specific Property Inquiry: Here, the student's queries relate to specific properties mentioned in previous chats. For instance, questions about cashback offers, guarantor requirements, or room types for a previously mentioned property fall under this category. In such scenarios, the 'properties' key should list the names of the properties being referenced, and other keys should be updated based on the chat context.\n2. General Property Availability: In this case, the student seeks more options for properties, typically around a university or within a city, without referring to a specific property. Here, the 'property' and 'property_manager' keys should be left empty, and other requirements should be updated accordingly, with most keys remaining empty except for 'university' and 'city', unless explicitly mentioned otherwise.\n\nAdditional Guidelines: 1. Always store the 'properties' key value as a list, regardless of the number of properties mentioned. 2. Ensure that the 'city' key remains up-to-date based on the current requirements, even if not explicitly stated. Additionally, if a student implies that accommodation is not needed in a specific city, set the 'city' key to an empty string. For example, if the student expresses a preference for properties near London Business School, it should be evident to update the 'city' key to 'London'. 3. Adjust the 'budget' key value based on student's current requirement. For example a student finds the previously recommended properties expensive, then the 'budget'key should store the cheapest of the prices of the previous properties accordinly. So in such cases note that 'budget' key will not store string 'affordable' but the actual cheapest price numeric value. Also budget could be specified as rent per week or per month. 4.  For guarantor key there are three possible values: a. If student is asking whether international guarantor is accepted then guarantor key will store 'INTERNATIONAL'. b. If student is asking for a local UK based guarantor then store 'LOCAL' as the value of guarantor key. c. if a student needs properties with no requirement of guarantor then store 'NONE' as the value of this key.' 5. If there are no queries regarding requirements, return `d={'properties':[], 'property_manager':'', 'city':'', 'university':'', 'room_type':'', 'budget':'', 'lease':'', 'guarantor':'','dual_occupancy':'','amenities':[]}`. 6. Correct any spelling errors in the key values. 7. Use a predefined list of property managers serving in the United Kingdom to update the 'property_manager' key.  We have following property managers serving in the United Kingdom:'iQ', 'Unite Students', 'Student Roost', 'HFS', 'Londonist', 'GoBritanya', 'FSL', 'Vita Student', 'CRM Students', 'Dwell Student', 'AXO', 'Collegiate', 'Cloud Student homes', 'True Student', 'Prime Student Living', 'Capitol Student', 'Iconinc', 'Mezzino', 'Liv Student', 'Allied', 'Premier Student Halls', 'Campus Living Villages', 'Study Inn', 'Student Castle', 'The Stay club', 'Hello Student', 'Downing', 'Canvas', 'Nurtur Student Living', 'DIGS Student', 'Abodus Students', 'Student beehive', 'X1 Lettings', 'Manor Villages', 'Student Cribs', 'Aspenhawk LTD', 'Project Student', 'City Block', 'Uni2 Rent', 'Bauhaus Student', 'Vivo Living', 'CA Ventures', 'Luxury Student Living', 'Uniplaces', 'Malden Hall', 'Future Generation', 'APPS Living'. Keep 'property_manager' key strictly empty if the requirement is unclear. 8. Dual occupany key can store only three types of value. 1. 'FREE':if the student is asking for free dual occupancy 2.'PAID':if the student is asking for paid dual occupancy. 3. 'ANY': if student is asking for just dual occupancy. Note that if student has no requirement regarding dual occupancy then keep the 'dual_occupancy' key empty. 9. Store the 'amenities' key as a list, capturing the preferences of students for various facilities. The values should be chosen strictly from the following list of amenities: ['free_breakfast', 'dinner_included', 'pet_friendly', 'no_university_no_pay', 'no_visa_no_pay', 'Private Bathroom', 'Private Kitchen', 'Private Room', 'WiFi', 'On-Site Gym', 'Fridge', 'Free Laundry', 'Utility Bills Included', 'Study desk and chair', 'Room Cleaning Services', 'Parking Space', 'Double Bed']. 10. Store the complete name of campus for 'university' key."
        )
    ]

    conversation_system.append(HumanMessage(content=conversation))

    content_keys = str(chat(conversation_system).content)

    flag = 0
    keys = ''
    for char in content_keys:
        if char == '{':
            flag = 1
        if char == '}':
            flag = 0
            keys += char
            break
        if flag == 1:
            keys += char

    # print("extracted keys: ",keys)
    keys = ast.literal_eval(keys)

    if keys['guarantor'] == 'INTERNATIONAL':
        keys[
            'guarantor'
        ] = 'The study of fundamental particles and their behavior at the quantum level'
    elif keys['guarantor'] == 'LOCAL':
        keys[
            'guarantor'
        ] = 'Agriculture is the science and practice of cultivating crops and raising livestock '
    elif keys['guarantor'] == 'NONE':
        keys[
            'guarantor'
        ] = 'Medical studies involve the comprehensive examination of human health, encompassing diagnosis, treatment'
    else:
        keys['guarantor'] = ''

    property_manager = keys['property_manager']

    if property_manager in const.updated_dictionary_with_tags:
        keys['property_manager'] += (
            ' '
            + const.updated_dictionary_with_tags[property_manager]['Language']
            + ': '
            + const.updated_dictionary_with_tags[property_manager]['Tags'][0]
            + ', '
            + const.updated_dictionary_with_tags[property_manager]['Tags'][1]
            + ', '
            + const.updated_dictionary_with_tags[property_manager]['Tags'][2]
        )

    budbuck = ['']

    if keys['dual_occupancy'] != '':
        if keys['dual_occupancy'] == 'FREE':
            keys['dual_occupancy'] = 'FREE Dual Occupancy'
        elif keys['dual_occupancy'] == 'PAID':
            keys['dual_occupancy'] = 'Dual Occupancy:'
        elif keys['dual_occupancy'] == 'ANY':
            keys['dual_occupancy'] = 'Dual Occupancy'

    # Convert the numbers to integers
    numbers = [int(num) for num in re.findall(r'\d+', str(keys['budget']))]
    # Find the maximum number

    if len(numbers) > 0:
        budget = max(numbers)

        budbuck = []
        bud_min = budget - 50
        if bud_min < 0:
            bud_min = 0
        for i in range(bud_min, budget, 50):
            budbuck.append(bucket_budget(i))

    properties = ''
    count = 0
    faq_data = ''
    t_0 = timeit.default_timer()

    if status == 'CONFIRMED':
        ## loading the files city
        # base_folder = '/Users/apple/Documents/Bot/Vectors/cities/'
        base_folder = Path.cwd() / 'vectors' / 'cities'
        closest_folder = find_closest_match(keys['city'], city_folder_names)
        closest_folder_path = os.path.join(base_folder, closest_folder)

        '''Add the path to faq pickle file containing embedding '''
        with open(os.path.join(closest_folder_path, 'faq.pkl'), 'rb') as f:
            faq_index = pickle.load(f)

        ## loading the files university

        base_folder = closest_folder_path
        uni_folder_names = [
            folder
            for folder in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, folder))
        ]

        closest_folder = find_closest_match(keys['university'], uni_folder_names)
        closest_folder_path = os.path.join(base_folder, closest_folder)

        '''Add the path to sales pickle file containing embedding '''
        with open(os.path.join(closest_folder_path, 'sales.pkl'), 'rb') as f:
            faiss_index = pickle.load(f)

            '''Add the path to supply config pickle file containing embedding '''
        with open(os.path.join(closest_folder_path, 'supply_config.pkl'), 'rb') as f:
            supply_config = pickle.load(f)

        # main_content="\nAnswer user's query using following data related to various properties:\n\n"
        main_content = "\nHere are details of the properties available based on the user's query. All the following properties are commissionable and are managed by different property managers. Closely look at who manages the property, we have the following property managers serving in United Kingdom: 'iQ', 'Unite Students', 'Student Roost', 'HFS', 'Londonist', 'GoBritanya', 'FSL', 'Vita Student', 'CRM Students', 'Dwell Student', 'AXO', 'Collegiate', 'Cloud Student homes', 'True Student', 'Prime Student Living', 'Capitol Student', 'Iconinc', 'Mezzino', 'Liv Student', 'Allied', 'Premier Student Halls', 'Campus Living Villages', 'Study Inn', 'Student Castle', 'The Stay club', 'Hello Student', 'Downing', 'Canvas', 'Nurtur Student Living', 'DIGS Student', 'Abodus Students', 'Student beehive', 'X1 Lettings', 'Manor Villages', 'Student Cribs', 'Aspenhawk LTD', 'Project Student', 'City Block', 'Uni2 Rent', 'Bauhaus Student', 'Vivo Living', 'CA Ventures', 'Luxury Student Living', 'Uniplaces', 'Malden Hall', 'Future Generation', 'APPS Living'. Recommend a list of properties and rooms with their structured details, along with the URL links for all properties. Do not create URLs on your own; provide the URLs for the properties available with us. If the user's university does not match exactly with the university in our database, avoid providing the distance from the wrong university, instead first look for the closest match university and ask the student if it is the university he/she meant. For example, if 'London School of Science' is asked, and in the database we have university 'London School of Commerce', then ask the student 'Did you mean London School of Commerce'. Provide recommendations based on these properties only. If a customer asks any questions about a property and theres no mention about it in the data below, like 'can i smoke in my room' then dont give any answers based on assumptions, simply ask the customer to visit the property page or Global Property Consultants for such FAQs. Or if a customer asks for a link to any particular property, provide them with the link and explain how to go to the property page, choose the room type, etc. Click on 'Enquire' to connect with the Global Property Consultant to complete the booking. Always look closely at the available intakes or move in for the properties. For example January 2023 intake means the student can move in the property from January 2023. Also strictly look at the guarantor requirements, we have three kind of properties: a. accept UK based local guarantor only, b. accepts international guarnator also, c. dont require any guarantor. For example if the student asks for INTERNATIONAL guarantor requirement and we have properties which accepts only UK based local guarantors then say that we dont have such properties who accept international guarantor.\n\nNOTES:\n1. While giving the URL, add 'https://' before the URL/link to the property's website.\n2. 'Distance_walking' is the distance between the property and the university.\n3. Prices are rents per week as well as per month and in the specific currency of the country. If prices are rent per month then changes to rent per week by dividing by the rent by 4, and then answer the user's query.\n4. If the property has dual occupancy rooms it will have Dual Occupancy tags under the Studio config. Dual occupancy means two students can stay together in single room. If there is no tag for dual occupancy it means that room does not offer dual occupancy. If dual occupancy tag is there, it can be of two types: 1. Free: Students are only required to cover the standard room rate, that is the price of this single studio config, with no additional fees. 2. Paid: For a supplementary fee, you can enjoy the benefits of dual occupancy, adding an extra layer of convenience to your stay.\n5. If there are no properties within student's low budget, simply say that we dont have rooms in this budget.\n\n"
        if isinstance(keys['properties'], list):
            if len(keys['properties']) > 0:
                for i, j in enumerate(range(len(keys['properties']))):
                    properties += ' property' + str(i + 1) + ':' + keys['properties'][i]

                    docs = faiss_index.similarity_search(
                        input
                        + ' '
                        + keys['university']
                        + ', '
                        + keys['city']
                        + keys['properties'][i]
                        + ' '
                        + keys['property_manager']
                        + ', '
                        + keys['guarantor']
                        + keys['room_type']
                        + ' '
                        + str(budbuck),
                        k=1,
                    )
                    faq_docs = faq_index.similarity_search(
                        input
                        + ' '
                        + keys['properties'][i]
                        + ' '
                        + keys['city']
                        + keys['property_manager'],
                        k=1,
                    )
                    for doc in docs:
                        main_content += (
                            correct(
                                property_manager_encoded(
                                    guarantor_fix(doc.page_content), 1
                                )
                            )
                            + '\n\n'
                        )

                    for doc in faq_docs:
                        faq_data += (
                            property_manager_encoded(doc.page_content, 2) + '\n\n'
                        )

                    remainder = int(6 / len(keys['properties'])) - 1

                    remainder_faq = int(6 / len(keys['properties'])) - 1
                    if remainder > 0:
                        docs = faiss_index.similarity_search(
                            input
                            + ' '
                            + keys['university']
                            + ', '
                            + keys['city']
                            + keys['properties'][i]
                            + ' '
                            + keys['property_manager']
                            + ', '
                            + keys['guarantor']
                            + keys['room_type']
                            + ' '
                            + str(budbuck),
                            k=remainder,
                        )
                        faq_docs = faq_index.similarity_search(
                            input
                            + ' '
                            + keys['properties'][i]
                            + ' '
                            + keys['city']
                            + keys['property_manager'],
                            k=remainder_faq,
                        )

                        for doc in docs:
                            main_content += (
                                correct(
                                    property_manager_encoded(
                                        guarantor_fix(doc.page_content), 1
                                    )
                                )
                                + '\n\n'
                            )

                        for doc in faq_docs:
                            faq_data += (
                                property_manager_encoded(doc.page_content, 2) + '\n\n'
                            )

            else:
                if keys['university'] != '':
                    count = 0

                    # print("ah chaleya 1 :",input," ",keys['university']+" "+keys['city']+" "+keys['property_manager']+" "+keys['guarantor']+keys['room_type']+" " +str(budbuck))
                    t_0_ = timeit.default_timer()
                    docs = supply_config.similarity_search(
                        input
                        + ' '
                        + keys['university']
                        + ' '
                        + keys['city']
                        + ' '
                        + keys['property_manager']
                        + ' '
                        + keys['guarantor']
                        + keys['room_type']
                        + ' '
                        + str(budbuck),
                        k=3000,
                    )
                    current_one = ['current']
                    for i, doc in enumerate(docs):
                        # if str.lower(keys['university']) in str.lower(doc.page_content):
                        if budbuck[0] in doc.page_content:
                            if str.lower(keys['dual_occupancy']) in str.lower(
                                doc.page_content
                            ):
                                flag = True
                                for y in keys['amenities']:
                                    if y not in doc.page_content:
                                        flag = False
                                if flag:
                                    # print("yes",i)
                                    if count < 6:
                                        # print("count:",count)
                                        if p_name(doc.page_content) not in current_one:
                                            current_one.append(p_name(doc.page_content))
                                            main_content += doc.page_content + '\n\n'
                                            count += 1
                                    else:
                                        break
                    t_1_ = timeit.default_timer()
                    print(t_1_ - t_0_, ' ehna')

                elif keys['city'] != '':
                    count = 0
                    # print("count:",count)
                    current_one = ['current']
                    ##print("ah chaleya 2:",input+" "+keys['university']+keys['city']+" "+properties+" "+keys['property_manager']+", "+keys['guarantor']+keys['room_type']+" "+str(budbuck))
                    docs = supply_config.similarity_search(
                        input
                        + ' '
                        + keys['university']
                        + keys['city']
                        + ' '
                        + properties
                        + ' '
                        + keys['property_manager']
                        + ', '
                        + keys['guarantor']
                        + keys['room_type']
                        + ' '
                        + str(budbuck),
                        k=3000,
                    )
                    for i, doc in enumerate(docs):
                        if str.lower(keys['city']) in str.lower(doc.page_content):
                            if budbuck[0] in doc.page_content:
                                if str.lower(keys['dual_occupancy']) in str.lower(
                                    doc.page_content
                                ):
                                    flag = True
                                    for y in keys['amenities']:
                                        if y not in doc.page_content:
                                            flag = False
                                    if flag:
                                        # print("yes",i)
                                        if count < 6:
                                            # print("count:",count)
                                            if (
                                                p_name(doc.page_content)
                                                not in current_one
                                            ):
                                                current_one.append(
                                                    p_name(doc.page_content)
                                                )
                                                main_content += (
                                                    doc.page_content + '\n\n'
                                                )
                                                count += 1
                                        else:
                                            break
                else:
                    docs = supply_config.similarity_search(
                        input
                        + ' '
                        + keys['university']
                        + keys['city']
                        + ' '
                        + keys['property_manager']
                        + ', '
                        + keys['guarantor']
                        + keys['room_type']
                        + ' '
                        + str(budbuck),
                        k=6,
                    )
                    for doc in docs:
                        main_content += doc.page_content + '\n\n'
                faq_docs = faq_index.similarity_search(
                    input + ' ' + keys['city'] + ' ' + keys['property_manager'], k=6
                )
                for doc in faq_docs:
                    faq_data += property_manager_encoded(doc.page_content, 2) + '\n\n'
        else:
            properties = keys['properties']
            if keys['city'] != '':
                count = 0
                # print("count:",count)
                print(
                    'ah chaleya 3:',
                    input
                    + ' '
                    + keys['university']
                    + keys['city']
                    + ' '
                    + properties
                    + ' '
                    + keys['property_manager']
                    + ', '
                    + keys['guarantor']
                    + keys['room_type']
                    + ' '
                    + str(budbuck),
                )
                docs = supply_config.similarity_search(
                    input
                    + ' '
                    + keys['university']
                    + keys['city']
                    + ' '
                    + properties
                    + ' '
                    + keys['property_manager']
                    + ', '
                    + keys['guarantor']
                    + keys['room_type']
                    + ' '
                    + str(budbuck),
                    k=3000,
                )
                current_one = ['current']
                for i, doc in enumerate(docs):
                    if str.lower(keys['city']) in str.lower(doc.page_content):
                        if budbuck[0] in doc.page_content:
                            if str.lower(keys['dual_occupancy']) in str.lower(
                                doc.page_content
                            ):
                                flag = True
                                for y in keys['amenities']:
                                    if y not in doc.page_content:
                                        flag = False
                                if flag:
                                    # print("yes",i)
                                    if count < 6:
                                        # print("count:",count)
                                        if p_name(doc.page_content) not in current_one:
                                            current_one.append(p_name(doc.page_content))
                                            main_content += doc.page_content + '\n\n'
                                            count += 1
                                    else:
                                        break
            else:
                docs = supply_config.similarity_search(
                    input
                    + ' '
                    + keys['university']
                    + keys['city']
                    + ' '
                    + properties
                    + ' '
                    + keys['property_manager']
                    + ', '
                    + keys['guarantor']
                    + keys['room_type']
                    + ' '
                    + str(budbuck),
                    k=6,
                )
                for doc in docs:
                    main_content += doc.page_content + '\n\n'

            faq_docs = faq_index.similarity_search(
                input
                + ' '
                + properties
                + ' '
                + keys['city']
                + ' '
                + keys['property_manager'],
                k=6,
            )

            for doc in faq_docs:
                faq_data += property_manager_encoded(doc.page_content, 2) + '\n\n'

        if count == 0:
            docs = supply_config.similarity_search(input, k=2)
            for doc in docs:
                main_content += doc.page_content + '\n\n'

        faq_docs = faq_index.similarity_search(input, k=2)
        for doc in faq_docs:
            faq_data += property_manager_encoded(doc.page_content, 2) + '\n\n'

        if len(keys['city']) > 0:
            docs = faiss_index.similarity_search(
                'Cheapest room property or minimum, rent room property or price in: '
                + keys['city']
                + ' ',
                k=10,
            )
            for doc in docs:
                if str.lower(keys['city']) in str.lower(doc.page_content):
                    main_content += doc.page_content + '\n\n'
                    break
            # for doc in docs:
            #     main_content += doc.page_content + "\n\n"
        # print("aaha ta done hogya")
        t_1 = timeit.default_timer()
        print(t_1 - t_0, ' pura')
        # main_content+="\nIf the provided room availability does not correspond with the student's inquiry or preferences, please inform the student of the alternative options and inquire whether the suggested accommodations meet their desired criteria. Additionally,
        main_content += 'If the student has not specified the city, university, or any other preferences, kindly ask for clarification. For instance, if the student is inquiring about rooms near BPP University, which has campuses in multiple cities, please prompt the student to specify the preferred city for their accommodation search. '
        if len(messages) > 1:
            # print("hun faq aala chalda")
            #         faq_base=[
            #         SystemMessage(
            #             content = "As an assistant for UniAcco, your role is to provide detailed information and assistance to students regarding student accommodation services. Address frequently asked questions (FAQs) related to properties available on UniAcco's website, ensuring your responses are solely based on the provided data. Be clear and accurate to enhance the students experience. Also try to address inquiries related to restaurants, Gurudwaras, temples, malls, student accommodation, cities, or localities in the United Kingdom. Do not speculate on room or property availability; if unable to assist, respond with 'False.' Additionally, inform students about two other UniAcco verticals: UniScholars (https://www.unischolars.com) for university applications and visa processes, and UniCreds (https://www.unicreds.com) for assistance in obtaining loans at favorable interest rates. Direct students to these links if they inquire about universities or loans. NOTE: 1. Keep in mind that there can be some FAQs that are applicable to that specific property, and some FAQs are applicable to all the properties managed by any property manager. 2. Strictly reply 'False' if the customer's query is asking for properties near any university. 3. When providing URLs, only provide the given URLs, dont create URL of your own.")]
            #         faq_base.append(HumanMessage(content=input+properties+" "+keys['city']+" "+property_manager))
            #         faq_base.append(SystemMessage(content=faq_data))
            #         faq_response=chat(faq_base).content
            #         if faq_response!="False" and contains_negative_words(faq_response)!=True:
            #             main_content+="\nIf the property details mentioned above do not address student's query then information provided below related to FAQs might help. So please refer to this Frequently Asked Questions (FAQs) section :"+faq_response
            #         print("faq_response:",faq_response)
            main_content += (
                "\n\nIf the property details mentioned above do not address student's query then information provided below related to FAQs might help. So please refer to this Frequently Asked Questions (FAQs) section :"
                + faq_data
            )
            # print("hun sales bible aala chalda")
        #         sales_bible_base=[
        #         SystemMessage(
        #             content = "Your task is to assist sales agents at UniAcco, a company specializing in providing student accommodations for those studying abroad. The company exclusively caters to students and lists properties in various countries and cities on its website, uniacco.com. The sales follow-up process is structured as follows: For Januray 2024 intake leads: Day 1: 2 calls + WA + Email (additional call and WhatsApp after 5-hour gap), Day 2: 1 call + WA/Email, Day 3: Rest day, Day 4: 2 calls + WA + Email, Day 5: Break, Day 6: 1 call + WA/Email, Day 7: Break, Day 8: 2 calls + WA/Email. For September 2024-25 intake leads: Day 1: 2 calls + WA + Email, Day 2: 1 call + WA/Email, Day 3-4: Break, Day 5: 2 calls + WA/Email, Day 6: Break, Day 7: 1 call + WA/Email, Day 8: Break, Day 9: 1 call + WA/Email, Day 10-11: Break, Day 12: 2 calls + WA + Email. Your role is to equip sales agents with essential information about UniAcco, including sales data, growth, bookings, and adherence guidelines. Additionally, if agents inquire about university applications or loans, direct them to UniScholars for university application and visa assistance, and to UniCreds for student loans. Furthermore, if agents ask about competitors, inform them that UniAcco faces competition from Amber Student, University Living, and UHomes, among others. Your responses should properly structured, concise(nearly 50 words) and strictly limited to the available information. If a query is about the availability of accommodations, rooms, or properties, or if the required information is not available, strictly respond with 'False'.")]
        #         sales_bible_base.append(HumanMessage(content=input+" "+keys['university']+" "+properties+" "+" "+property_manager+","+keys['room_type']+" "+keys['city'] +str(budbuck)+" "))
        #         docs=sales_bible.similarity_search(input+" "+keys['university']+" "+properties+" "+" "+property_manager+","+keys['room_type']+" "+keys['city'] +str(budbuck)+" ", k=2)
        #         sales_bible_data=""
        #         sales_bible_data = "\n\n".join(doc.page_content for doc in docs)
        #         print("sales_bible: ",sales_bible_data)
        #         sales_bible_base.append(SystemMessage(content=sales_bible_data))
        #         print("sales_bible_base: ",sales_bible_base)
        #         sales_bible_response=chat(sales_bible_base).content
        #         print("sales_bible_response: ",sales_bible_response)
        #         if sales_bible_response!="False" and contains_negative_words(sales_bible_response)!=True:
        #             main_content+="\Information related to sales:"+sales_bible_response

        # main_content+="\n\nUse following information related to sales adherence at UniAcco:"+" The sales follow-up process is structured as follows: For Januray 2024 intake leads: Day 1: 2 calls + WA + Email (additional call and WhatsApp after 5-hour gap), Day 2: 1 call + WA/Email, Day 3: Rest day, Day 4: 2 calls + WA + Email, Day 5: Break, Day 6: 1 call + WA/Email, Day 7: Break, Day 8: 2 calls + WA/Email. For September 2024-2025 intake leads: Day 1: 2 calls + WA + Email, Day 2: 1 call + WA/Email, Day 3-4: Break, Day 5: 2 calls + WA/Email, Day 6: Break, Day 7: 1 call + WA/Email, Day 8: Break, Day 9: 1 call + WA/Email, Day 10-11: Break, Day 12: 2 calls + WA + Email. "
        #     additional_system=[
        #     SystemMessage(
        #         content = "Provide detailed information for inquiries only related to restaurants, Gurudwaras, temples, malls, student accommodation, cities, or localities in the United Kingdom. If there is no relevant information available or if you cannot assist, or if the query is related to availability of accomodations or rooms or properties, respond strictly only 'False'. NOTE: Use the information that we have two other verticals for assisting students going abroad for studies:\n1. UniScholars: Help students throughout their university application and visa process. [UniScholars](https://www.unischolars.com)\n2. UniCreds: Help students in getting loans at the best interest rates. [UniCreds](https://www.unicreds.com).\n Give students the link to these verticals if they ask about universities  or loans respectively."
        # )
        #     ]
        #     additional_system.append(HumanMessage(content=input+" "+keys['university']+" "+properties+" "+" "+property_manager+","+keys['room_type']+" "+keys['city'] +str(budbuck)+" "))
        #     additional_response=chat(additional_system).content
        #     if additional_response!="False" and contains_negative_words(additional_response)!=True:
        #         main_content+="\nAdditional information: To be used only if it is true and pertinent to above given supply data, otherwise don't use this information and say that we dont cater in this city or country, we cater only in United Kingdom.  :"+additional_response
        #             print(keys)
        #             print(main_content)
        # print("additional_response: ",additional_response)
        #         print("Aaha auna chihda: ",properties)
        # print(messages)
        messages.append(SystemMessage(content='\n' + main_content + '\n'))
        if len(messages) >= 3:
            messages.append(
                HumanMessage(
                    content=input
                    + properties
                    + ' '
                    + keys['city']
                    + ' '
                    + property_manager
                    + ' '
                    + keys['university']
                )
            )
            if keys['city'] != '':
                messages.append(
                    SystemMessage(
                        content='\n\nIf a student has already chosen the city as '
                        + keys['city']
                        + ", always thoroughly review all the data and respond to queries in the following ways: 1. If asked about properties or rooms near any university, recommend a list of properties sorted in ascending order based on distances in miles (mi). Provide necessary details along with all the links to the properties. For distances, only consider the distance if the correct property and university names match. Avoid providing false distances based on similar names of universities or properties. For example, if asked for 'London School Of Business,' do not give the distance for 'London Business School.' If there are no correct matches available, indicate that the data is not available."
                    )
                )
        # print(messages)
        t_0_ = timeit.default_timer()
        ai_response = chat(messages).content
        t_1_ = timeit.default_timer()
        print(t_1_ - t_0_, ' ehna ai response lainda')
        # print(str(ai_response))
        messages.pop()
        if len(messages) >= 3:
            messages.pop()
            messages.pop()
            messages.append(HumanMessage(content=input))

        #     messages.append(HumanMessage(content=input))
        messages.append(AIMessage(content=ai_response))
        # print(keys)
        archive.append(datetime.now())
        message_history.append({'role': 'assistant', 'content': f'{ai_response}'})

        # get pairs of msg["content"] from message history, skipping the pre-prompt: here.
        response = [
            (message_history[i]['content'], message_history[i + 1]['content'])
            for i in range(0, len(message_history) - 1, 2)
        ]  # convert to tuples of list
        # print(response)
        return response, status, messages, message_history
    else:
        main_content = ''
        if 'university' in keys['university']:
            keys['university'] = keys['university'].replace('university', '')

        if keys['city'] == '':
            main_content += 'Student has not chosen the city'
        else:
            main_content += (
                'The student has indicated their choice of city as,' + keys['city']
            )
        if keys['university'] == '':
            main_content += '. Student has not chosen the university'
        else:
            # base_folder = '/Users/apple/Documents/Bot/Vectors/cities/'
            base_folder = Path.cwd() / 'vectors' / 'cities'
            closest_folder = find_closest_match(keys['city'], city_folder_names)
            closest_folder_path = os.path.join(base_folder, closest_folder)
            base_folder = closest_folder_path
            uni_folder_names = [
                folder
                for folder in os.listdir(base_folder)
                if os.path.isdir(os.path.join(base_folder, folder))
            ]

            main_content += '. and mentioned the university as, ' + keys['university']
            main_content += (
                '. However, we need to verify this information. If confirmation has not been obtained earlier, please share the following names with the student and confirm which university or campus in London they are specifically referring to. '
                + keys['city']
                + ': '
            )
            for i, j in enumerate(uni_folder_names):
                if str.lower(keys['university']) in j:
                    main_content += '\n' + j

        messages.append(HumanMessage(content=input))
        messages.append(SystemMessage(content=main_content))
        # print(messages)
        ai_response = chat(messages).content
        messages_ = []
        if ai_response == 'CONFIRMED':
            ai_response = "Thank you for confirming your city and university! We're excited to help you find the perfect accommodation. To assist you better, could you please provide some details about your preferences(like Budget, Lease duration, Room type, or any amenity?) Your input will help us tailor our recommendations to meet your needs."
            status = 'CONFIRMED'
            messages_ = [
                SystemMessage(
                    content="UniAcco specializes in providing student accommodations for those studying abroad. Our properties are exclusively available for students and not for non-students or individuals on a casual trip. Currently, our listings cover various properties in different countries and cities, with a current focus on the United Kingdom. As a seasoned sales executive, your primary goal is to engage potential customers and successfully complete the sale by aligning their needs with our available options.\n\nThe process involves three steps: Step 1 - Inquire about their city and university, Step 2 - Confirm the information gathered in Step 1, and Step 3 - Recommend suitable properties based on specific requirements such as room types, budget range, lease terms, intake, and desired amenities.\n\nThe current status is Step 1 and Step 2 are complete, and we are currently working on Step 3. Your task now is to recommend properties based on the identified requirements. Begin the conversation with a welcoming message (e.g., Hi, welcome to UniAcco) while also asking about their specific accommodation needs.\n\nImportant Notes:\n1. Stay focused on the conversation and steer towards completing the sale.\n2. Politely state that questions outside the scope of the discussion are 'Out of Scope,' especially those unrelated to UniAcco or sensitive topics.\n3. Keep responses concise and to the point.\n4. If a customer expresses interest in a city or country where we don't currently have properties, clearly communicate that we currently do not offer accommodations in that location.\n5. Maintain a positive tone when addressing questions related to UniAcco or UniAcco's website."
                )
            ]
            for i, j in enumerate(message_history):
                if i % 2 == 0:
                    messages_.append(HumanMessage(content=j['content']))
                else:
                    messages_.append(AIMessage(content=j['content']))
            # messages=messages_[:-1]
        #     messages.append(HumanMessage(content=input))
        messages.pop()
        messages.append(AIMessage(content=ai_response))
        # print(keys)
        archive.append(datetime.now())
        message_history.append({'role': 'assistant', 'content': f'{ai_response}'})

        # get pairs of msg["content"] from message history, skipping the pre-prompt: here.
        response = [
            (message_history[i]['content'], message_history[i + 1]['content'])
            for i in range(0, len(message_history) - 1, 2)
        ]  # convert to tuples of list

        return response, status, messages, message_history


def get_response(input, msg_history, status_):
    query = input

    global messages
    global status
    global message_history
    messages = [
        SystemMessage(
            content="UniAcco specializes in providing student accommodations for students going abroad for studies. Our properties are not for non-students or any one going for trip, we only have rooms for students. On our websites we list various properties only in United Kingdom. I want you to act like an experienced sales executive and talk to our potential customers with the sole motive of completing the sale and matching the demand with the supply. The process is divided into three steps: Step 1: Ask them about their city and university. Step 2: Confirming the step 1 requirement. Step 3: Recommend properties based on the requirements like room types and even the budget range, lease, intake and the amenities they are looking for.\n\n Status of Step 1: In progress. Step 2: In progress. Step 3: Not started. So as of now we will focus on step 1 and 2, and insist student to do the needful. Once student shared his university requirement, then we need to confirm the student's actual university(or may be campus as there are multiple campuses sometimes) and city, this will help map the correct city and university name from our database. So provide the list of closest match universities and ask the student to confirm the university or campus. Once the student confirms city and university requirements, then strictly reply 'CONFIRMED'. You can begin the conversation with welcome message(Hi, welcome to UniAcco-along with this asking about their requirements). Important Note: 1. Do not go astray from the conversation, and converge to a sale. 2. Answer politely that the questions is strictly 'Out of Scope' for irrelevant questions(like who is Vin Diesel or ELon Musk, or book me a flight) or questions that can cause harm to UniAcco public reputation or any sensitive topic. 3. Answer should be in concise and to the point. 4. If during step 1, student's wants room near a university which is not in United Kingdom, or wants property in some other country than UK, strictly say that we currently do not have supply.5. Always try to answer positively about questions related to UniAcco or UniAcoo's website. If there are questions wanting to connect to customer support, please ask them to connect to customer support at +44 808 501 5198 or contact@uniacco.com.\n\nNotes: We have two other verticals for assisting students going abroad for studies:\n1. UniScholars: Help students throughout their university application and visa process. [UniScholars](https://www.unischolars.com)\n2. UniCreds: Help students in getting loans at the best interest rates. [UniCreds](https://www.unicreds.com).\n\n"
        )
    ]

    status = status_
    if status != 'CONFIRMED':
        messages = [
            SystemMessage(
                content="UniAcco specializes in providing student accommodations for students going abroad for studies. Our properties are not for non-students or any one going for trip, we only have rooms for students. On our websites we list various properties only in United Kingdom. I want you to act like an experienced sales executive and talk to our potential customers with the sole motive of completing the sale and matching the demand with the supply. The process is divided into three steps: Step 1: Ask them about their city and university. Step 2: Confirming the step 1 requirement. Step 3: Recommend properties based on the requirements like room types and even the budget range, lease, intake and the amenities they are looking for.\n\n Status of Step 1: In progress. Step 2: In progress. Step 3: Not started. So as of now we will focus on step 1 and 2, and insist student to do the needful. Once student shared his university requirement, then we need to confirm the student's actual university(or may be campus as there are multiple campuses sometimes) and city, this will help map the correct city and university name from our database. So provide the list of closest match universities and ask the student to confirm the university or campus. Once the student confirms city and university requirements, then strictly reply 'CONFIRMED'. You can begin the conversation with welcome message(Hi, welcome to UniAcco-along with this asking about their requirements). Important Note: 1. Do not go astray from the conversation, and converge to a sale. 2. Answer politely that the questions is strictly 'Out of Scope' for irrelevant questions(like who is Vin Diesel or ELon Musk, or book me a flight) or questions that can cause harm to UniAcco public reputation or any sensitive topic. 3. Answer should be in concise and to the point. 4. If during step 1, student's wants room near a university which is not in United Kingdom, or wants property in some other country than UK, strictly say that we currently do not have supply.5. Always try to answer positively about questions related to UniAcco or UniAcoo's website. If there are questions wanting to connect to customer support, please ask them to connect to customer support at +44 808 501 5198 or contact@uniacco.com.\n\nNotes: We have two other verticals for assisting students going abroad for studies:\n1. UniScholars: Help students throughout their university application and visa process. [UniScholars](https://www.unischolars.com)\n2. UniCreds: Help students in getting loans at the best interest rates. [UniCreds](https://www.unicreds.com).\n\n"
            )
        ]
    else:
        messages = [
            SystemMessage(
                content="UniAcco specializes in providing student accommodations for those studying abroad. Our properties are exclusively available for students and not for non-students or individuals on a casual trip. Currently, our listings cover various properties in different countries and cities, with a current focus on the United Kingdom. As a seasoned sales executive, your primary goal is to engage potential customers and successfully complete the sale by aligning their needs with our available options.\n\nThe process involves three steps: Step 1 - Inquire about their city and university, Step 2 - Confirm the information gathered in Step 1, and Step 3 - Recommend suitable properties based on specific requirements such as room types, budget range, lease terms, intake, and desired amenities.\n\nThe current status is Step 1 and Step 2 are complete, and we are currently working on Step 3. Your task now is to recommend properties based on the identified requirements. Begin the conversation with a welcoming message (e.g., Hi, welcome to UniAcco) while also asking about their specific accommodation needs.\n\nImportant Notes:\n1. Stay focused on the conversation and steer towards completing the sale.\n2. Politely state that questions outside the scope of the discussion are 'Out of Scope,' especially those unrelated to UniAcco or sensitive topics.\n3. Keep responses concise and to the point.\n4. If a customer expresses interest in a city or country where we don't currently have properties, clearly communicate that we currently do not offer accommodations in that location.\n5. Maintain a positive tone when addressing questions related to UniAcco or UniAcco's website. If there are questions wanting to connect to customer support, please ask them to connect to customer support at +44 808 501 5198 or contact@uniacco.com.\n\nNotes: We have two other verticals for assisting students going abroad for studies:\n1. UniScholars: Help students throughout their university application and visa process. [UniScholars](https://www.unischolars.com)\n2. UniCreds: Help students in getting loans at the best interest rates. [UniCreds](https://www.unicreds.com).\n\n"
            )
        ]

    for i, message in enumerate(msg_history):
        if i > 0:
            if i % 2 == 0:  # Check if the index is even
                messages.append(HumanMessage(content=str(message['content'])))
            else:
                messages.append(AIMessage(content=str(message['content'])))

    message_history = msg_history

    # print(query)
    def process_input(query, messages, status, message_history):
        # Assuming 'predict' is a function that takes the user's query and returns a response, status, and updated messages
        response, status, messages, message_history = predict(
            query, status, messages, message_history
        )

        if status == 'CONFIRMED':
            messages = [
                SystemMessage(
                    content="UniAcco specializes in providing student accommodations for those studying abroad. Our properties are exclusively available for students and not for non-students or individuals on a casual trip. Currently, our listings cover various properties in different countries and cities, with a current focus on the United Kingdom. As a seasoned sales executive, your primary goal is to engage potential customers and successfully complete the sale by aligning their needs with our available options.\n\nThe process involves three steps: Step 1 - Inquire about their city and university, Step 2 - Confirm the information gathered in Step 1, and Step 3 - Recommend suitable properties based on specific requirements such as room types, budget range, lease terms, intake, and desired amenities.\n\nThe current status is Step 1 and Step 2 are complete, and we are currently working on Step 3. Your task now is to recommend properties based on the identified requirements. Begin the conversation with a welcoming message (e.g., Hi, welcome to UniAcco) while also asking about their specific accommodation needs.\n\nImportant Notes:\n1. Stay focused on the conversation and steer towards completing the sale.\n2. Politely state that questions outside the scope of the discussion are 'Out of Scope,' especially those unrelated to UniAcco or sensitive topics.\n3. Keep responses concise and to the point.\n4. If a customer expresses interest in a city or country where we don't currently have properties, clearly communicate that we currently do not offer accommodations in that location.\n5. Maintain a positive tone when addressing questions related to UniAcco or UniAcco's website. If there are questions wanting to connect to customer support, please ask them to connect to customer support at +44 808 501 5198 or contact@uniacco.com.\n\nNotes: We have two other verticals for assisting students going abroad for studies:\n1. UniScholars: Help students throughout their university application and visa process. [UniScholars](https://www.unischolars.com)\n2. UniCreds: Help students in getting loans at the best interest rates. [UniCreds](https://www.unicreds.com).\n\n"
                )
            ] + messages[1:]
        return (
            response,
            messages,
            status,
            message_history,
        )  # Return the response to be displayed in the chatbot

    (
        final_resp,
        updated_messages,
        updated_status,
        updated_message_history,
    ) = process_input(query, messages, status, message_history)
    messages = updated_messages
    status = updated_status
    message_history = updated_message_history
    user, bot = final_resp[-1]
    return bot, message_history, status


# if __name__ == "__main__":
#     msg_his,status = input_message()
#     print(msg_his,"\n",status)
