import ast
import os
import pickle
import re
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


def create_vectors():
    p_faq = pd.read_csv(
        'https://redash.uniacco.com/api/queries/1853/results.csv?api_key=GMyGIWf88thY6lcM89v05Nt9elj9ddYzWwgCe9aq'
    )
    m_faq = pd.read_csv(
        'https://redash.uniacco.com/api/queries/1852/results.csv?api_key=Q5yboiJBVFIjqa6SbNaZCpmM7BmZxewqfb9h9s4R'
    )

    df1 = pd.read_csv(
        'https://redash.uniacco.com/api/queries/1323/results.csv?api_key=MKESa1oXYiWLatDvYrxqjS5yyBCoeW3XjS9TQPvs'
    )
    df2 = pd.read_csv(
        'https://redash.uniacco.com/api/queries/2154/results.csv?api_key=JYlkViPaUv9sXjYBdzZF7mqrT7kPPUSstYvX9YzN'
    )
    if df1.shape[0] != df2.shape[0]:
        return 'Data shape error'

    df1[
        [
            'Rent_weekly_or_monthly',
            'currency',
            'deposit',
            'security_deposit',
            'is_available',
            'room_type',
            'dual_occupancy',
            'config_tags',
            'room_type_config',
            'available_from',
            'guarantor',
            'plan_1',
            'plan_2',
            'plan_3',
            'plan_4',
            'URL',
        ]
    ] = df2[
        [
            'Rent_weekly_or_monthly',
            'currency',
            'deposit',
            'security_deposit',
            'is_available',
            'room_type',
            'dual_occupancy',
            'config_tags',
            'room_type_config',
            'available_from',
            'guarantor',
            'plan_1',
            'plan_2',
            'plan_3',
            'plan_4',
            'URL',
        ]
    ]

    df = df1

    df = df.drop_duplicates()
    df = df[df['country_id'] == 1]
    df = df[df['univ_city_id'] == df['prop_city_id']]

    df1 = pd.read_csv(
        'https://redash.uniacco.com/api/queries/1440/results.csv?api_key=Gx4segzB7nga0xnDISi0PRq9M8gZWSZK4LxvjCiy'
    )
    df2 = pd.read_csv(
        'https://redash.uniacco.com/api/queries/1487/results.csv?api_key=1bBwawdlwx1ZGWNgSjyW2Avaglcp6ROGpQ610BXW'
    )

    df['university_name'] = df['university_name'].str.replace("'", '')
    df['distance'] = df['distance_walking'].apply(lambda x: str(x)[:-2])
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    print('Done 1')
    prop_amenities = {}
    for i, j in enumerate(df2['property_id'].unique()):
        prop_conf = df2[(df2['property_id'] == j)]
        # | (df2['property_idprop_config_id'] == "-")]
        prop_amenities[j] = []
        if sum(prop_conf['kind'] == 'APARTMENT') > 0:
            prop_amenities[j] = {}
            kind_prop_conf = prop_conf[prop_conf['kind'] == 'APARTMENT']
            prop_amenities[j]['APARTMENT'] = list(kind_prop_conf['name1'].unique())

        if sum(prop_conf['kind'] == 'COMMUNITY') > 0:
            prop_amenities[j] = {}
            kind_prop_conf = prop_conf[prop_conf['kind'] == 'COMMUNITY']
            prop_amenities[j]['COMMUNITY'] = list(kind_prop_conf['name1'].unique())

    prop_conf_amenities = {}
    for i, j in enumerate(df2['prop_config_id'].unique()):
        if j != '-':
            prop_conf = df2[(df2['prop_config_id'] == j)]

            if sum(prop_conf['kind'] == 'CONFIG') > 0:
                kind_prop_conf = prop_conf[prop_conf['kind'] == 'CONFIG']
                prop_conf_amenities[j] = list(kind_prop_conf['name1'])

    d = {}
    for i, j in enumerate(df['university_name'].unique()):
        uni = df[df['university_name'] == j]
        d[j] = []
        uni = uni.sort_values(['distance'])
        for a, b in enumerate(uni['id'].unique()):
            pro = uni[uni['id'] == b]
            d[j].append([b, list(pro['distance'])[0]])

    distance_category = []
    for i, (j, k, l) in enumerate(zip(df['id'], df['university_name'], df['distance'])):
        if k is not np.nan:
            mini = d[k][0][1]
            if len(d[k]) <= 5:
                sixth = d[k][len(d[k]) - 1][1]
            else:
                sixth = d[k][5][1]
            if l <= sixth:
                distance_category.append('Closest or Nearest')
            else:
                distance_category.append('Near')
        else:
            distance_category.append('')

    df['distance_category'] = distance_category
    print('Done 2')
    df1['self_provided'] = df1['self_provided'].apply(
        lambda x: '--UniAcco Exclusive--' if x == 1 else '--Property offers--'
    )

    co1 = {}
    for i, j in enumerate(df1['country_id'].unique()):
        country = df1[
            (df1['country_id'] == j)
            & (df1['manager_id'].isna())
            & (df1['prop_id'].isna())
        ]
        country['cashback'] = (
            country['self_provided'] + '.' + country['title'] + '.' + country['details']
        )
        co1[j] = str(country['cashback'].values)

    man1 = {}
    for i, j in enumerate(df1['manager_id'].unique()):
        manag = df1[(df1['manager_id'] == j) & (df1['prop_id'].isna())]

        for a, b in enumerate(manag['country_id'].unique()):
            man1[j] = {}
            manager = manag[(manag['country_id'] == b)]
            if len(manager) > 0:
                manager['cashback'] = (
                    manager['self_provided']
                    + '.'
                    + manager['title']
                    + '.'
                    + manager['details']
                )
                man1[j][b] = str(manager['cashback'].values)
        manag = manag[(manag['country_id'].isna())]
        if len(manag) > 0:
            manag['cashback'] = (
                manag['self_provided'] + '.' + manag['title'] + '.' + manag['details']
            )
            man1[j]['all'] = str(manag['cashback'].values)
    print('Done 3')

    prop1 = {}
    for i, j in enumerate(df1['prop_id'].unique()):
        propertee = df1[df1['prop_id'] == j]
        if len(propertee) > 0:
            propertee['cashback'] = (
                propertee['self_provided']
                + '.'
                + propertee['title']
                + '.'
                + propertee['details']
            )
            prop1[j] = str(propertee['cashback'].values)

    offers = []
    for i, (j, k, l) in enumerate(zip(df['id'], df['manag_id'], df['country_id'])):
        offers.append(co1[l])
        if j in prop1:
            offers[i] = offers[i] + str(prop1[j])
        if k in man1:
            if l in man1[k]:
                offers[i] = offers[i] + str(man1[k][l])
            else:
                offers[i] = offers[i] + str(man1[k]['all'])

    from bs4 import BeautifulSoup

    # Function to remove <p> and </p> tags
    def remove_p_tags(offer):
        soup = BeautifulSoup(offer, 'html.parser')
        return ''.join(str(tag) for tag in soup.find_all(text=True))

    # Remove <p> and </p> tags from each offer
    cleaned_offers = [remove_p_tags(offer) for offer in offers]
    df['offers'] = cleaned_offers
    df['offers'] = df['offers'].str.strip('[]')

    def split_string_by_newline(text):
        return [s.strip() for s in text.split('\n') if s.strip()]

    # Apply the function to the 'offers' column
    df['offers'] = df['offers'].apply(split_string_by_newline)

    def split_text(text):
        text_list = ' '.join(text).split('--UniAcco Exclusive--')
        return [t.strip() for t in text_list if t.strip()]

    df['offers'] = df['offers'].apply(split_text)
    print('Done 4')
    plan_ = {}
    for i, j in enumerate(df['prop_name'].unique()):
        plans = df[df['prop_name'] == j].iloc[0]  # Use .iloc[0] to get the first row
        plan1 = str(plans['plan_1'])
        plan2 = str(plans['plan_2'])  # Assuming you meant 'plan_2' instead of 'plan_1'
        plan3 = str(plans['plan_3'])  # Assuming you meant 'plan_3' instead of 'plan_1'
        plan4 = str(plans['plan_4'])  # Assuming you meant 'plan_4' instead of 'plan_1'
        instalments = ''
        monthly_quarterly = ''

        if plan1 != 'nan':
            if len(plan1) > 4:
                if len(monthly_quarterly) > 0:
                    monthly_quarterly += ' or ' + plan1
                else:
                    monthly_quarterly += ' ' + plan1
            else:
                if len(instalments) > 0:
                    instalments += ' or ' + plan1
                else:
                    instalments += ' ' + plan1
        if plan2 != 'nan':
            if len(plan2) > 4:
                if len(monthly_quarterly) > 0:
                    monthly_quarterly += ' or ' + plan2
                else:
                    monthly_quarterly += ' ' + plan2
            else:
                if len(instalments) > 0:
                    instalments += ' or ' + plan2
                else:
                    instalments += ' ' + plan2
        if plan3 != 'nan':
            if len(plan1) > 4:
                if len(monthly_quarterly) > 0:
                    monthly_quarterly += ' or ' + plan3
                else:
                    monthly_quarterly += ' ' + plan3
            else:
                if len(instalments) > 0:
                    instalments += ' or ' + plan3
                else:
                    instalments += ' ' + plan3
        if plan4 != 'nan':
            if len(plan1) > 4:
                if len(monthly_quarterly) > 0:
                    monthly_quarterly += ' or ' + plan4
                else:
                    monthly_quarterly += ' ' + plan4
            else:
                if len(instalments) > 0:
                    instalments += ' or ' + plan4
                else:
                    instalments += ' ' + plan4
        #     print(j, plan1, plan2, plan3, plan4,monthly_quarterly,instalments)
        if monthly_quarterly != '':
            if instalments != '':
                plan_[j] = (
                    'You can pay either in',
                    instalments,
                    ' instalments or either,',
                    monthly_quarterly,
                )
            else:
                plan_[j] = 'You can pay in', monthly_quarterly, ' instalments.'
        elif instalments != '':
            plan_[j] = 'You can pay in', instalments, ' instalments.'
        else:
            plan_[j] = ''
        ip = ''
        for ips in plan_[j]:
            ip += ips
        plan_[j] = ip

    # grntr={
    #     'UK_BASED': 'Agriculture is the science and practice of cultivating crops and raising livestock ',
    #     'LOCAL': 'Local Required',
    #     'INTERNATIONAL': 'The study of fundamental particles and their behavior at the quantum level',
    #     'NOT_REQUIRED': 'Medical studies involve the comprehensive examination of human health, encompassing diagnosis, treatment',
    # }

    grntr = {
        'UK_BASED': 'LOCAL UK BASED ONLY',
        'LOCAL': 'Local Required',
        'INTERNATIONAL': 'International Accepted.',
        'NOT_REQUIRED': 'Not Required',
    }

    intakess = list(df['available_from'].unique())
    sorted_intake_list = sorted(
        intakess, key=lambda x: (int(x.split()[1]), x.split()[0])
    )

    updated_dictionary_with_tags = {
        'iQ': {
            'Language': 'French',
            'Tags': [
                'Romance language',
                'The Little Prince by Antoine de Saint-Exupéry',
                'Gendered nouns',
            ],
        },
        'Unite Students': {
            'Language': 'German',
            'Tags': [
                'Compound words',
                'Faust by Johann Wolfgang von Goethe',
                'Four noun case system',
            ],
        },
        'Student Roost': {
            'Language': 'Spanish',
            'Tags': [
                'Verb conjugations',
                'Don Quixote by Miguel de Cervantes',
                'Two distinct "to be" verbs',
            ],
        },
        'HFS': {
            'Language': 'Italian',
            'Tags': [
                'Musical terminology',
                'The Divine Comedy by Dante Alighieri',
                'Double consonants',
            ],
        },
        'Londonist': {
            'Language': 'Dutch',
            'Tags': [
                'Afrikaans similarity',
                'Diary of Anne Frank by Anne Frank',
                'Devoicing of final consonants',
            ],
        },
        'GoBritanya': {
            'Language': 'Portuguese',
            'Tags': [
                'Brazil and Portugal varieties',
                'The Lusiads by Luís de Camões',
                'Nasal vowels',
            ],
        },
        'FSL': {
            'Language': 'Russian',
            'Tags': [
                'Cyrillic alphabet',
                'War and Peace by Leo Tolstoy',
                'Aspect of verbs',
            ],
        },
        'Vita Student': {
            'Language': 'Chinese',
            'Tags': [
                'Tonal language',
                'Journey to the West by Wu Cheng\'en',
                'Characters instead of alphabet',
            ],
        },
        'CRM Students': {
            'Language': 'Japanese',
            'Tags': [
                'Three writing systems',
                'The Tale of Genji by Murasaki Shikibu',
                'Politeness levels in speech',
            ],
        },
        'Dwell Student': {
            'Language': 'Korean',
            'Tags': [
                'Hangul script',
                'The Cloud Dream of the Nine by Kim Man-jung',
                'Subject-Object-Verb order',
            ],
        },
        'AXO': {
            'Language': 'Arabic',
            'Tags': [
                'Abjad script',
                'One Thousand and One Nights',
                'Root and pattern system',
            ],
        },
        'Collegiate': {
            'Language': 'Turkish',
            'Tags': [
                'Agglutinative language',
                'Museum of Innocence by Orhan Pamuk',
                'Vowel harmony',
            ],
        },
        'Cloud Student homes': {
            'Language': 'Hindi',
            'Tags': [
                'Devanagari script',
                'Godan by Premchand',
                'Postpositions instead of prepositions',
            ],
        },
        'True Student': {
            'Language': 'Swedish',
            'Tags': [
                'North Germanic language',
                'Pippi Longstocking by Astrid Lindgren',
                'Definite articles as suffixes',
            ],
        },
        'Prime Student Living': {
            'Language': 'Greek',
            'Tags': [
                'Ancient and modern forms',
                'The Odyssey by Homer',
                'Three genders in nouns',
            ],
        },
        'Capitol Student': {
            'Language': 'Danish',
            'Tags': [
                'Stød (glottal stop)',
                'Fairy Tales by Hans Christian Andersen',
                'En/et indefinite articles',
            ],
        },
        'Iconinc': {
            'Language': 'Norwegian',
            'Tags': [
                'Two official forms: Bokmål and Nynorsk',
                'Hunger by Knut Hamsun',
                'Pitch accent',
            ],
        },
        'Mezzino': {
            'Language': 'Finnish',
            'Tags': [
                'Uralic family, not Indo-European',
                'The Kalevala by Elias Lönnrot',
                'No articles, definite or indefinite',
            ],
        },
        'Liv Student': {
            'Language': 'Polish',
            'Tags': [
                'Slavic language',
                'Pan Tadeusz by Adam Mickiewicz',
                'Seven cases for nouns',
            ],
        },
        'Allied': {
            'Language': 'Czech',
            'Tags': [
                'West Slavic language',
                'The Good Soldier Švejk by Jaroslav Hašek',
                'Aspect as a grammatical category',
            ],
        },
        'Premier Student Halls': {
            'Language': 'Hungarian',
            'Tags': [
                'Non-Indo-European',
                'The Tragedy of Man by Imre Madách',
                'Agglutinative grammar',
            ],
        },
        'Campus Living Villages': {
            'Language': 'Romanian',
            'Tags': [
                'Latin-derived vocabulary',
                'Mihai Eminescu, national poet',
                'Definite articles as enclitics',
            ],
        },
        'Study Inn': {
            'Language': 'Bulgarian',
            'Tags': [
                'Cyrillic alphabet',
                'Under the Yoke by Ivan Vazov',
                'No infinitive verb form',
            ],
        },
        'Student Castle': {
            'Language': 'Hebrew',
            'Tags': [
                'Ancient and modern revival',
                'The Bible, Tanakh',
                'Root-based word construction',
            ],
        },
        'The Stay club': {
            'Language': 'Thai',
            'Tags': [
                'Tone marks',
                'The Tale of Khun Chang Khun Phaen, folk epic',
                'Subject-verb-object order',
            ],
        },
        'Hello Student': {
            'Language': 'Vietnamese',
            'Tags': [
                'Latin alphabet with diacritics',
                'The Tale of Kieu by Nguyễn Du',
                'Tonal and monosyllabic',
            ],
        },
        'Downing': {
            'Language': 'Indonesian',
            'Tags': [
                'Malayo-Polynesian language',
                'This Earth of Mankind by Pramoedya Ananta Toer',
                'No tense, gender, or plurals',
            ],
        },
        'Canvas': {
            'Language': 'Filipino',
            'Tags': [
                'Tagalog-based',
                'Noli Me Tangere by José Rizal',
                'Verb-subject-object order',
            ],
        },
        'Nurtur Student Living': {
            'Language': 'Ukrainian',
            'Tags': [
                'Cyrillic alphabet',
                'Kobzar by Taras Shevchenko',
                'Seven noun cases',
            ],
        },
        'DIGS Student': {
            'Language': 'Slovak',
            'Tags': [
                'West Slavic language',
                'Sirocco by Jozef Ignác Bajza',
                'Inflected language',
            ],
        },
        'Abodus Students': {
            'Language': 'Estonian',
            'Tags': [
                'Uralic family, similar to Finnish',
                'Truth and Justice by A. H. Tammsaare',
                'No future tense, no gender',
            ],
        },
        'Student beehive': {
            'Language': 'Latvian',
            'Tags': [
                'Baltic language',
                "The Fisherman's Son by Vilis Lācis",
                'Two genders, seven cases',
            ],
        },
        'X1 Lettings': {
            'Language': 'Lithuanian',
            'Tags': [
                'Oldest Indo-European language',
                'The Forest of the Gods by Balys Sruoga',
                'Preserved archaic features',
            ],
        },
        'Manor Villages': {
            'Language': 'Malay',
            'Tags': [
                'Austronesian language',
                'Salina by A. Samad Said',
                'Agglutinative, no inflection',
            ],
        },
        'Student Cribs': {
            'Language': 'Serbian',
            'Tags': [
                'Cyrillic and Latin scripts',
                'The Bridge on the Drina by Ivo Andrić',
                'Ejective sounds',
            ],
        },
        'Aspenhawk LTD': {
            'Language': 'Croatian',
            'Tags': [
                'South Slavic language',
                'The Return of Philip Latinowicz by Miroslav Krleža',
                'Seven cases for nouns',
            ],
        },
        'Project Student': {
            'Language': 'Slovenian',
            'Tags': [
                'South Slavic language',
                'Baptism on the Savica by France Prešeren',
                'Dual grammatical number',
            ],
        },
        'City Block': {
            'Language': 'Bosnian',
            'Tags': [
                'South Slavic language',
                'Death and the Dervish by Meša Selimović',
                'Ijekavian pronunciation',
            ],
        },
        'Uni2 Rent': {
            'Language': 'Icelandic',
            'Tags': [
                'North Germanic language',
                'The Sagas of Icelanders',
                'Inflectional grammar',
            ],
        },
        'Bauhaus Student': {
            'Language': 'Welsh',
            'Tags': [
                'Celtic language',
                'Mabinogion, medieval prose',
                'Initial consonant mutation',
            ],
        },
        'Vivo Living': {
            'Language': 'Irish',
            'Tags': [
                'Gaelic language',
                'Ulysses by James Joyce',
                'Verb-subject-object syntax',
            ],
        },
        'CA Ventures': {
            'Language': 'Maltese',
            'Tags': [
                'Semitic language in Latin script',
                'In the Eye of the Sun by Alex Vella Gera',
                'Phonetic spelling',
            ],
        },
        'Luxury Student Living': {
            'Language': 'Afrikaans',
            'Tags': [
                'Derived from Dutch',
                "Fiela's Child by Dalene Matthee",
                'Double negative feature',
            ],
        },
        'Uniplaces': {
            'Language': 'Swahili',
            'Tags': [
                'Bantu language',
                'Utendi wa Tambuka, epic poem',
                'Noun class system',
            ],
        },
        'Malden Hall': {
            'Language': 'Albanian',
            'Tags': [
                'Indo-European but standalone branch',
                'The General of the Dead Army by Ismail Kadare',
                'Unique lexical terms',
            ],
        },
        'Future Generation': {
            'Language': 'Macedonian',
            'Tags': [
                'South Slavic language',
                'Tales of Olden Times by Grigor Prlicev',
                'Complex verb system',
            ],
        },
        'APPS Living': {
            'Language': 'Georgian',
            'Tags': [
                'Kartvelian language',
                "The Knight in the Panther's Skin by Shota Rustaveli",
                'Unique alphabet',
            ],
        },
    }

    manager_encode = []
    for i, j in enumerate(df['manager']):
        if j in updated_dictionary_with_tags:
            manager_encode.append(
                updated_dictionary_with_tags[j]['Language']
                + ': '
                + updated_dictionary_with_tags[j]['Tags'][0]
                + ', '
                + updated_dictionary_with_tags[j]['Tags'][1]
                + ', '
                + updated_dictionary_with_tags[j]['Tags'][2]
            )
        else:
            manager_encode.append('-')
    df['manager_encode'] = manager_encode

    # Assuming df is your DataFrame
    df['dual_occupancy'] = df['dual_occupancy'].fillna('')

    def categorize_price(row):
        if row['rent_weekly'] < 200:
            return 'Low budget/ Cheap'
        elif 200 <= row['rent_weekly'] <= 300:
            return 'Budget friendly'
        else:
            return 'Premium Expensive'

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
        words = [
            '0',
            '50',
            '100',
            '150',
            '200',
            '250',
            '300',
            '350',
            '400',
            '450',
            '500',
            '550',
            '600',
            '650',
            '700',
            '750',
            '800',
            '850',
            '900',
            '950',
            '1000',
            '1050',
            '1100',
            '1150',
            '1200',
            '1250',
            '1300',
            '1350',
            '1400',
            '1450',
            '1500',
            '1550',
            '1600',
            '1650',
            '1700',
            '1750',
            '1800',
            '1850',
            '1900',
            '1950',
            '2000',
        ]

        if budget > 2000:
            return 'Budget_Excess_Opulent_Option'

        bucket_label = words[int(budget / 50)]
        bucket_label += '_to_' + words[int(budget / 50) + 1]

        return 'Budget_tag: ' + budget_buckets[bucket_label]

    print('Done 5')

    ## main run

    '''enter your openai api key'''
    os.environ['OPENAI_API_KEY'] = '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'
    '''Add the path to your pdf file'''
    # Assuming you have a DataFrame named city_df with a column 'city_name'
    # Example DataFrame creation (replace this with your actual DataFrame)

    # Specify the full path for the 'cities' folder
    base_folder = Path.cwd() / 'vectors'
    cities_folder = os.path.join(base_folder, 'cities')

    # Create the 'cities' folder if it doesn't exist
    os.makedirs(cities_folder, exist_ok=True)

    # Enumerate through the city names and create a folder for each city
    for a, b in enumerate(df['city'].unique()):
        city_folder_names = ['delhi']
        city_name = str.lower(b)
        if city_name not in (city_folder_names):
            city_df = df[df['city'] == b]

            city_folder_path = os.path.join(cities_folder, city_name)

            # Create the city folder if it doesn't exist
            os.makedirs(city_folder_path, exist_ok=True)

            # Save the current standard output (e.g., the console)

            original_stdout = sys.stdout

            # Define the file where you want to save the document
            output_file = os.path.join(city_folder_path, 'faq_document.txt')

            # Open the file in write mode and set it as the new standard output
            with open(output_file, 'w') as file_:
                sys.stdout = file_

                ## FAQ print statements to create faq_document.txt

                ## FAQ print statements to create faq_document.txt

                for i, j in enumerate(city_df['id'].unique()):
                    mngr = list(city_df[city_df['id'] == j]['manager_encode'])[0]
                    maangr = list(city_df[city_df['id'] == j]['manager'])[0]
                    prpty = list(city_df[city_df['id'] == j]['prop_name'])[0]
                    prpty_city = list(city_df[city_df['id'] == j]['city'])[0]
                    link = list(city_df[city_df['id'] == j]['URL'])[0]
                    # print("FAQ: property:",prpty,",city:",prpty_city,",property manager:",mngr,".",)

                    prop_faq = p_faq[p_faq['prop_id'] == j]
                    if len(prop_faq) > 0:
                        prop_faq.drop_duplicates()
                        for a, question_answer in enumerate(
                            prop_faq['question_answer']
                        ):
                            print(
                                'FAQ specific to this property:',
                                prpty,
                                ',city:',
                                prpty_city,
                                ',property manager:',
                                maangr,
                                mngr,
                                '.',
                                question_answer,
                                '. Link to property',
                                link,
                                '.\n',
                            )

                    manager_faq = m_faq[m_faq['name'] == maangr]
                    if len(manager_faq) > 0:
                        manager_faq.drop_duplicates()
                        for a, question_answer in enumerate(
                            manager_faq['question_answer']
                        ):
                            print(
                                'FAQ valid for all properties managed by this manager. Property:',
                                prpty,
                                ',city:',
                                prpty_city,
                                ',property manager:',
                                maangr,
                                mngr,
                                '.',
                                question_answer,
                                '. Link to property',
                                link,
                                '.\n',
                            )

                for i, j in enumerate(city_df['manager'].unique()):
                    mngr = list(city_df[city_df['manager'] == j]['manager_encode'])[0]
                    maangr = list(city_df[city_df['manager'] == j]['manager'])[0]
                    if list(city_df[city_df['manager'] == j]['allow_payments'])[0] == 0:
                        print(
                            'FAQ valid for all properties managed by this manager. property manager:',
                            maangr,
                            mngr,
                            '. QUESTION: Can UniAcco handle the processing of payments or deposits on behalf of students, or does the manager prefer direct payments to them? ANSWER: No, the manager does not permit payments to be made to UniAcco. Instead, the manager prefers payments to be made directly to them.\n',
                        )
                    else:
                        print(
                            'FAQ valid for all properties managed by this manager. property manager:',
                            maangr,
                            mngr,
                            '. QUESTION: Can UniAcco handle the processing of payments or deposits on behalf of students, or does the manager prefer direct payments to them? ANSWER: Yes, the manager allows UniAcco to take payments or deposits from students.\n',
                        )

            # Restore the original standard output
            sys.stdout = original_stdout

            # Inform the user that the output has been saved
            print(f'FAQ Output has been saved to {output_file}')

            ## ethee
            # Specify the file path
            file_path = (
                output_file  # Replace 'your_file.txt' with the path to your text file
            )

            try:
                # Open the file for reading
                with open(file_path, 'r') as file:
                    # Read the entire contents of the file
                    file_contents = file.read()

                # Now, 'file_contents' contains the text from the file
            #     print(file_contents)

            except FileNotFoundError:
                print(f"The file '{file_path}' was not found.")
            except Exception as e:
                print(f'An error occurred: {str(e)}')

            raw_text = file_contents
            '''Divide the input data into chunks
                This will help in reducing the embedding size as we will see in the code
                as well as reduce the token size for the query,'''
            text_splitter = CharacterTextSplitter(
                separator='\n',
                chunk_size=300,
                chunk_overlap=20,
                length_function=len,
            )
            texts = text_splitter.split_text(raw_text)

            embeddings = OpenAIEmbeddings(
                model='text-embedding-3-small',
                deployment='text-embedding-3-small',
                disallowed_special=(),
            )
            docsearch = FAISS.from_texts(texts, embeddings)

            faq_pickle = os.path.join(city_folder_path, 'faq.pkl')
            with open(faq_pickle, 'wb') as f:
                pickle.dump(docsearch, f)
            ## ethee

            for p, q in enumerate(city_df['university_name'].unique()):
                university_name = str.lower(q)
                univ_df = city_df[city_df['university_name'] == q]

                univ_folder_path = os.path.join(city_folder_path, university_name)

                # Create the city folder if it doesn't exist
                os.makedirs(univ_folder_path, exist_ok=True)

                ### sales output document

                # Save the current standard output (e.g., the console)
                original_stdout = sys.stdout

                # Define the file where you want to save the document
                output_file = os.path.join(univ_folder_path, 'output_document.txt')

                # Open the file in write mode and set it as the new standard output
                with open(output_file, 'w') as file_:
                    sys.stdout = file_

                    def number_to_excel_column(n):
                        result = ''
                        while n > 0:
                            n, remainder = divmod(n - 1, 26)
                            result = chr(65 + remainder) + result
                        return result

                    for a, b in enumerate(univ_df['country'].unique()):
                        country = univ_df[univ_df['country'] == b]
                        country['rent_weekly'] = country.apply(
                            lambda row: row['rent']
                            if row['Rent_weekly_or_monthly'] == 'WEEK'
                            else row['rent'] / 4,
                            axis=1,
                        )
                        country['price_range'] = country.apply(categorize_price, axis=1)

                        for m, n in enumerate(country['city'].unique()):
                            city = country[country['city'] == n]
                            uni_dist_min_price_prop = {}
                            for i, j in enumerate(city['prop_name'].unique()):
                                df_uni_min_price_prop = city[city['prop_name'] == j]
                                df_uni_min_price_prop = df_uni_min_price_prop[
                                    df_uni_min_price_prop['rent_weekly']
                                    == df_uni_min_price_prop['rent_weekly'].min()
                                ]
                                uni_dist_min_price_prop[j] = (
                                    str(list(df_uni_min_price_prop['rent'])[0])
                                    + ' '
                                    + str(list(df_uni_min_price_prop['currency'])[0])
                                    + ' per '
                                    + str(
                                        list(
                                            df_uni_min_price_prop[
                                                'Rent_weekly_or_monthly'
                                            ]
                                        )[0]
                                    )
                                )
                            city_min = city['rent_weekly'].min()

                            city_min = city[city['rent_weekly'] == city_min]
                            city_min = city_min[
                                ['prop_name', 'city', 'country', 'rent_weekly', 'URL']
                            ]
                            city_min = city_min.drop_duplicates()
                            minimum_price_props = list(univ_df.prop_name.unique())
                            minimum_pirce_props_dict = {}

                            for z, x in enumerate(city['university_name'].unique()):
                                uni_dist = city[city['university_name'] == x]
                                uni_dist = uni_dist.sort_values(['distance'])
                                uni_dist = uni_dist[
                                    [
                                        'prop_name',
                                        'city',
                                        'country',
                                        'distance_walking',
                                        'university_name',
                                        'URL',
                                        'distance',
                                        'distance_category',
                                    ]
                                ]
                                uni_dist = uni_dist.drop_duplicates()

                                for d1, (
                                    d_prop_name,
                                    d_city,
                                    d_country,
                                    d_distance_walking,
                                    d_university_name,
                                    d_URL,
                                    d_distance,
                                    d_distance_category,
                                ) in enumerate(
                                    zip(
                                        uni_dist['prop_name'],
                                        uni_dist['city'],
                                        uni_dist['country'],
                                        uni_dist['distance_walking'],
                                        uni_dist['university_name'],
                                        uni_dist['URL'],
                                        uni_dist['distance'],
                                        uni_dist['distance_category'],
                                    )
                                ):
                                    prop = city[city['prop_name'] == d_prop_name]
                                    intakess = list(prop['available_from'].unique())
                                    sorted_intake_list = sorted(
                                        intakess,
                                        key=lambda x: (int(x.split()[1]), x.split()[0]),
                                    )

                                    offs = list(prop['offers'])[0]
                                    offer_prop = ''
                                    for z, x in enumerate(offs):
                                        if z > 0:
                                            offer_prop += str(z) + ' ' + offs[z]

                                    prop_ = prop[
                                        [
                                            'dual_occupancy',
                                            'manager_encode',
                                            'manager',
                                            'lease',
                                            'rent',
                                            'Rent_weekly_or_monthly',
                                            'currency',
                                            'deposit',
                                            'security_deposit',
                                            'room_type',
                                            'room_type_config',
                                            'available_from',
                                            'rent_weekly',
                                            'guarantor',
                                        ]
                                    ]

                                    # Group by the specified columns and aggregate the values
                                    grouped_prop = (
                                        prop_.groupby(
                                            [
                                                'manager_encode',
                                                'manager',
                                                'room_type',
                                                'room_type_config',
                                                'lease',
                                                'available_from',
                                            ]
                                        )
                                        .agg(
                                            {
                                                'dual_occupancy': 'first',
                                                'rent': 'first',  # Use the appropriate aggregation function (e.g., 'mean', 'sum', etc.)
                                                'Rent_weekly_or_monthly': 'first',
                                                'rent_weekly': 'first',
                                                'currency': 'first',  # Assuming it's the same for all rows within a group
                                                'deposit': 'first',  # Use the appropriate aggregation function
                                                'security_deposit': 'first',  # Use the appropriate aggregation function
                                                'guarantor': 'first',
                                            }
                                        )
                                        .reset_index()
                                    )

                                    prop_ = grouped_prop
                                    prop_ = prop_.sort_values(['rent_weekly'])
                                    prop_ = prop_.drop_duplicates()

                                    prop_configs = ''
                                    for it, (
                                        dual_occ,
                                        manager_enc,
                                        manager,
                                        lease,
                                        rent,
                                        Rent_weekly_or_monthly,
                                        currency,
                                        deposit,
                                        security_deposit,
                                        room_type,
                                        room_type_config,
                                        available_from,
                                        rent_weekly,
                                        guarantor,
                                    ) in enumerate(
                                        zip(
                                            prop_['dual_occupancy'],
                                            prop_['manager_encode'],
                                            prop_['manager'],
                                            prop_['lease'],
                                            prop_['rent'],
                                            prop_['Rent_weekly_or_monthly'],
                                            prop_['currency'],
                                            prop_['deposit'],
                                            prop_['security_deposit'],
                                            prop_['room_type'],
                                            prop['room_type_config'],
                                            prop_['available_from'],
                                            prop_['rent_weekly'],
                                            prop_['guarantor'],
                                        )
                                    ):
                                        #                    if it==len(prop_)-1:
                                        #                         prop_configs+="Config "+str(it+1)+":"+" Room type "+str(room_type)+" - "+room_type_config+". "+bucket_budget(rent_weekly)+". Lease "+str(lease)+" weeks, intake "+str(available_from)+" for rent "+str(rent)+" "+str(currency)+" per "+str(Rent_weekly_or_monthly)+". Require advance deposit "+str(prop['deposit'].min())+str(currency)+" and security deposit "+str(prop['security_deposit'].min())+" "+str(currency)+". "
                                        #                     prop_configs+="Config "+str(it+1)+":"+" Room type "+str(room_type)+" - "+room_type_config+". "+bucket_budget(rent_weekly)+". Lease "+str(lease)+" weeks, intake "+str(available_from)+" for rent "+str(rent)+" "+str(currency)+" per "+str(Rent_weekly_or_monthly)+"."
                                        if it == 0:
                                            for q, w in enumerate(
                                                prop_['room_type'].unique()
                                            ):
                                                rt = prop_[prop_['room_type'] == w]
                                                prop_configs += (
                                                    'Configurations in room type:'
                                                    + str(q + 1)
                                                    + '.'
                                                    + str(w)
                                                    + ' are following:'
                                                )
                                                for e, r in enumerate(
                                                    rt['room_type_config'].unique()
                                                ):
                                                    rtc = rt[
                                                        rt['room_type_config'] == r
                                                    ]
                                                    prop_configs += (
                                                        'Config '
                                                        + chr(e + 65)
                                                        + ': '
                                                        + r
                                                        + '- '
                                                        + list(rtc['dual_occupancy'])[0]
                                                        + '.'
                                                    )
                                                    for t, y in enumerate(
                                                        rtc['lease'].unique()
                                                    ):
                                                        rtc_l = rtc[rtc['lease'] == y]
                                                        prop_configs += (
                                                            'Sub-config '
                                                            + str(t + 1)
                                                            + ': Lease '
                                                            + str(y)
                                                            + ' weeks. '
                                                        )
                                                        for f, g in enumerate(
                                                            rtc_l[
                                                                'available_from'
                                                            ].unique()
                                                        ):
                                                            rtc_l_int = rtc_l[
                                                                rtc_l['available_from']
                                                                == g
                                                            ]
                                                            prop_configs += (
                                                                bucket_budget(
                                                                    int(
                                                                        list(
                                                                            rtc_l_int[
                                                                                'rent_weekly'
                                                                            ]
                                                                        )[0]
                                                                    )
                                                                )
                                                                + ', '
                                                                + str(g)
                                                                + ' for rent '
                                                                + str(
                                                                    list(
                                                                        rtc_l_int[
                                                                            'rent'
                                                                        ]
                                                                    )[0]
                                                                )
                                                                + ' '
                                                                + str(
                                                                    str(
                                                                        list(
                                                                            rtc_l_int[
                                                                                'currency'
                                                                            ]
                                                                        )[0]
                                                                    )
                                                                )
                                                                + ' per '
                                                                + str(
                                                                    str(
                                                                        list(
                                                                            rtc_l_int[
                                                                                'Rent_weekly_or_monthly'
                                                                            ]
                                                                        )[0]
                                                                    )
                                                                )
                                                                + ' .'
                                                            )
                                            # if d_prop_name=="iQ City":
                                            #    print(prop_configs,"\n\n")
                                            prop_configs += (
                                                ' This property require a minimum advance deposit '
                                                + str(prop['deposit'].min())
                                                + str(list(prop['currency'])[0])
                                                + ' and minimum security deposit '
                                                + str(prop['security_deposit'].min())
                                                + ' '
                                                + str(list(prop['currency'])[0])
                                                + '. '
                                            )
                                    #                 if d_prop_name=="Mannequin House":
                                    #                     print("Distance category(",d_distance_category,"property properties rooms)",d_university_name,",",d_city,",",d_country,", distance from the university to property ",d_prop_name," is ",d_distance_walking,". Starting price or cheapest room of this property start is, ",uni_dist_min_price_prop[d_prop_name],".",prop_configs,".Guarrantor :",grntr[guarantor],". Installment plan :",plan_[j],"Cashbacks and offers :",offer_prop,"Link to the property, URL:"+list(prop['URL'])[0])
                                    print(
                                        'Distance category(',
                                        d_distance_category,
                                        'property or rooms). University: ',
                                        d_university_name,
                                        ',',
                                        d_city,
                                        ',',
                                        d_country,
                                        ', distance from property',
                                        d_prop_name,
                                        ',',
                                        d_city,
                                        ' is ',
                                        d_distance_walking,
                                        '. This property is managed by property manager',
                                        manager,
                                        manager_enc,
                                        '. Starting price or cheapest room of this property starts at, ',
                                        uni_dist_min_price_prop[d_prop_name],
                                        '. Guarrantor :',
                                        grntr[guarantor],
                                        '.Installment plan :',
                                        plan_[j],
                                        '. Cashbacks and offers :',
                                        offer_prop,
                                        '. Available for :',
                                        sorted_intake_list,
                                        '.',
                                        prop_configs,
                                        'Link to the property, URL:'
                                        + list(prop['URL'])[0],
                                    )
                                    if (
                                        d_prop_name not in minimum_pirce_props_dict
                                        and d_prop_name in minimum_price_props
                                    ):
                                        minimum_pirce_props_dict[d_prop_name] = (
                                            'Starting price or cheapest room of this property starts at, '
                                            + str(uni_dist_min_price_prop[d_prop_name])
                                            + '. This property is managed by property manager '
                                            + manager
                                            + ' '
                                            + '. Guarrantor :'
                                            + str(grntr[guarantor])
                                            + '. Installment plan :'
                                            + str(plan_[j])
                                            + 'Cashbacks and offers :'
                                            + str(offer_prop)
                                            + '. Available for :'
                                            + str(sorted_intake_list)
                                            + '.'
                                            + str(prop_configs)
                                            + 'Link to the property, URL:'
                                            + str(list(prop['URL'])[0])
                                        )
                            for e, (p, ci, co, r, u) in enumerate(
                                zip(
                                    city_min['prop_name'],
                                    city_min['city'],
                                    city_min['country'],
                                    city_min['rent_weekly'],
                                    city_min['URL'],
                                )
                            ):
                                print(
                                    'Cheapest room property or minimum, rent room property or price in:',
                                    ci,
                                    ',',
                                    co,
                                    ' is available ',
                                    p,
                                    ' ',
                                    minimum_pirce_props_dict[p],
                                )

                # Restore the original standard output
                sys.stdout = original_stdout

                # Inform the user that the output has been saved
                print(f'Output Document has been saved to {output_file}')

                ## ethee
                # Specify the file path
                file_path = output_file  # Replace 'your_file.txt' with the path to your text file

                try:
                    # Open the file for reading
                    with open(file_path, 'r') as file:
                        # Read the entire contents of the file
                        file_contents = file.read()

                    # Now, 'file_contents' contains the text from the file
                #     print(file_contents)

                except FileNotFoundError:
                    print(f"The file '{file_path}' was not found.")
                except Exception as e:
                    print(f'An error occurred: {str(e)}')

                raw_text = file_contents
                '''Divide the input data into chunks
                    This will help in reducing the embedding size as we will see in the code
                    as well as reduce the token size for the query,'''
                text_splitter = CharacterTextSplitter(
                    separator='\n',
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len,
                )
                texts = text_splitter.split_text(raw_text)

                embeddings = OpenAIEmbeddings(
                    model='text-embedding-3-small',
                    deployment='text-embedding-3-small',
                    disallowed_special=(),
                )
                docsearch = FAISS.from_texts(texts, embeddings)

                sales_pickle = os.path.join(univ_folder_path, 'sales.pkl')
                with open(sales_pickle, 'wb') as f:
                    pickle.dump(docsearch, f)
                ## ethee

                ### supply config wise output document

                original_stdout = sys.stdout

                # Define the file where you want to save the document
                output_file = os.path.join(
                    univ_folder_path, 'output_document_config.txt'
                )

                # Open the file in write mode and set it as the new standard output
                with open(output_file, 'w') as file_:
                    sys.stdout = file_

                    def number_to_excel_column(n):
                        result = ''
                        while n > 0:
                            n, remainder = divmod(n - 1, 26)
                            result = chr(65 + remainder) + result
                        return result

                    for a, b in enumerate(univ_df['country'].unique()):
                        country = univ_df[univ_df['country'] == b]
                        country['rent_weekly'] = country.apply(
                            lambda row: row['rent']
                            if row['Rent_weekly_or_monthly'] == 'WEEK'
                            else row['rent'] / 4,
                            axis=1,
                        )
                        country['price_range'] = country.apply(categorize_price, axis=1)

                        for m, n in enumerate(country['city'].unique()):
                            city = country[country['city'] == n]
                            uni_dist_min_price_prop = {}
                            for i, j in enumerate(city['prop_name'].unique()):
                                df_uni_min_price_prop = city[city['prop_name'] == j]
                                df_uni_min_price_prop = df_uni_min_price_prop[
                                    df_uni_min_price_prop['rent_weekly']
                                    == df_uni_min_price_prop['rent_weekly'].min()
                                ]
                                uni_dist_min_price_prop[j] = (
                                    str(list(df_uni_min_price_prop['rent'])[0])
                                    + ' '
                                    + str(list(df_uni_min_price_prop['currency'])[0])
                                    + ' per '
                                    + str(
                                        list(
                                            df_uni_min_price_prop[
                                                'Rent_weekly_or_monthly'
                                            ]
                                        )[0]
                                    )
                                )
                            city_min = city['rent_weekly'].min()

                            city_min = city[city['rent_weekly'] == city_min]
                            city_min = city_min[
                                ['prop_name', 'city', 'country', 'rent_weekly', 'URL']
                            ]
                            city_min = city_min.drop_duplicates()
                            minimum_price_props = list(univ_df.prop_name.unique())
                            minimum_pirce_props_dict = {}

                            for z, x in enumerate(city['university_name'].unique()):
                                uni_dist = city[city['university_name'] == x]
                                uni_dist = uni_dist.sort_values(['distance'])
                                uni_dist = uni_dist[
                                    [
                                        'prop_name',
                                        'city',
                                        'country',
                                        'distance_walking',
                                        'university_name',
                                        'URL',
                                        'distance',
                                        'distance_category',
                                    ]
                                ]
                                uni_dist = uni_dist.drop_duplicates()

                                for d1, (
                                    d_prop_name,
                                    d_city,
                                    d_country,
                                    d_distance_walking,
                                    d_university_name,
                                    d_URL,
                                    d_distance,
                                    d_distance_category,
                                ) in enumerate(
                                    zip(
                                        uni_dist['prop_name'],
                                        uni_dist['city'],
                                        uni_dist['country'],
                                        uni_dist['distance_walking'],
                                        uni_dist['university_name'],
                                        uni_dist['URL'],
                                        uni_dist['distance'],
                                        uni_dist['distance_category'],
                                    )
                                ):
                                    prop = city[city['prop_name'] == d_prop_name]
                                    intakess = list(prop['available_from'].unique())
                                    sorted_intake_list = sorted(
                                        intakess,
                                        key=lambda x: (int(x.split()[1]), x.split()[0]),
                                    )

                                    offs = list(prop['offers'])[0]
                                    offer_prop = ''
                                    for z, x in enumerate(offs):
                                        if z > 0:
                                            offer_prop += str(z) + ' ' + offs[z]

                                    prop_ = prop[
                                        [
                                            'id',
                                            'property_tags',
                                            'config_tags',
                                            'dual_occupancy',
                                            'manager_encode',
                                            'manager',
                                            'lease',
                                            'rent',
                                            'Rent_weekly_or_monthly',
                                            'currency',
                                            'deposit',
                                            'security_deposit',
                                            'room_type',
                                            'config_id',
                                            'room_type_config',
                                            'available_from',
                                            'rent_weekly',
                                            'guarantor',
                                        ]
                                    ]

                                    # Group by the specified columns and aggregate the values
                                    grouped_prop = (
                                        prop_.groupby(
                                            [
                                                'manager_encode',
                                                'manager',
                                                'id',
                                                'room_type',
                                                'config_id',
                                                'room_type_config',
                                                'lease',
                                                'available_from',
                                            ]
                                        )
                                        .agg(
                                            {
                                                'dual_occupancy': 'first',
                                                'rent': 'first',  # Use the appropriate aggregation function (e.g., 'mean', 'sum', etc.)
                                                'Rent_weekly_or_monthly': 'first',
                                                'rent_weekly': 'first',
                                                'currency': 'first',  # Assuming it's the same for all rows within a group
                                                'deposit': 'first',  # Use the appropriate aggregation function
                                                'security_deposit': 'first',  # Use the appropriate aggregation function
                                                'guarantor': 'first',
                                                'property_tags': 'first',
                                                'config_tags': 'first',
                                            }
                                        )
                                        .reset_index()
                                    )

                                    prop_ = grouped_prop
                                    prop_ = prop_.sort_values(['rent_weekly'])
                                    prop_ = prop_.drop_duplicates()

                                    for it, (
                                        dual_occ,
                                        manager_enc,
                                        manager,
                                        lease,
                                        rent,
                                        Rent_weekly_or_monthly,
                                        currency,
                                        deposit,
                                        security_deposit,
                                        room_type,
                                        room_type_config,
                                        available_from,
                                        rent_weekly,
                                        guarantor,
                                    ) in enumerate(
                                        zip(
                                            prop_['dual_occupancy'],
                                            prop_['manager_encode'],
                                            prop_['manager'],
                                            prop_['lease'],
                                            prop_['rent'],
                                            prop_['Rent_weekly_or_monthly'],
                                            prop_['currency'],
                                            prop_['deposit'],
                                            prop_['security_deposit'],
                                            prop_['room_type'],
                                            prop['room_type_config'],
                                            prop_['available_from'],
                                            prop_['rent_weekly'],
                                            prop_['guarantor'],
                                        )
                                    ):
                                        #                    if it==len(prop_)-1:
                                        #                         prop_configs+="Config "+str(it+1)+":"+" Room type "+str(room_type)+" - "+room_type_config+". "+bucket_budget(rent_weekly)+". Lease "+str(lease)+" weeks, intake "+str(available_from)+" for rent "+str(rent)+" "+str(currency)+" per "+str(Rent_weekly_or_monthly)+". Require advance deposit "+str(prop['deposit'].min())+str(currency)+" and security deposit "+str(prop['security_deposit'].min())+" "+str(currency)+". "
                                        #                     prop_configs+="Config "+str(it+1)+":"+" Room type "+str(room_type)+" - "+room_type_config+". "+bucket_budget(rent_weekly)+". Lease "+str(lease)+" weeks, intake "+str(available_from)+" for rent "+str(rent)+" "+str(currency)+" per "+str(Rent_weekly_or_monthly)+"."
                                        if it == 0:
                                            for q, w in enumerate(
                                                prop_['room_type'].unique()
                                            ):
                                                rt = prop_[prop_['room_type'] == w]

                                                for e, r in enumerate(
                                                    rt['room_type_config'].unique()
                                                ):
                                                    prop_configs = (
                                                        '. The recommended room config is a '
                                                        + r
                                                    )
                                                    # " of room kind "+str(w)
                                                    rtc = rt[
                                                        rt['room_type_config'] == r
                                                    ]
                                                    prop_configs += (
                                                        list(rtc['dual_occupancy'])[0]
                                                        + '.'
                                                    )
                                                    for t, y in enumerate(
                                                        rtc['lease'].unique()
                                                    ):
                                                        rtc_l = rtc[rtc['lease'] == y]
                                                        prop_configs += (
                                                            ' Sub-config '
                                                            + str(t + 1)
                                                            + ': Lease '
                                                            + str(y)
                                                            + ' weeks. '
                                                        )
                                                        for f, g in enumerate(
                                                            rtc_l[
                                                                'available_from'
                                                            ].unique()
                                                        ):
                                                            rtc_l_int = rtc_l[
                                                                rtc_l['available_from']
                                                                == g
                                                            ]

                                                            prop_configs += (
                                                                bucket_budget(
                                                                    int(
                                                                        list(
                                                                            rtc_l_int[
                                                                                'rent_weekly'
                                                                            ]
                                                                        )[0]
                                                                    )
                                                                )
                                                                + ', '
                                                                + str(g)
                                                                + ' for rent '
                                                                + str(
                                                                    list(
                                                                        rtc_l_int[
                                                                            'rent'
                                                                        ]
                                                                    )[0]
                                                                )
                                                                + ' '
                                                                + str(
                                                                    str(
                                                                        list(
                                                                            rtc_l_int[
                                                                                'currency'
                                                                            ]
                                                                        )[0]
                                                                    )
                                                                )
                                                                + ' per '
                                                                + str(
                                                                    str(
                                                                        list(
                                                                            rtc_l_int[
                                                                                'Rent_weekly_or_monthly'
                                                                            ]
                                                                        )[0]
                                                                    )
                                                                )
                                                            )
                                                            # print(str(rtc_l_int['security_deposit'].values[0]))

                                                    prop_configs += (
                                                        '. This room requires an advance deposit '
                                                        + str(
                                                            int(list(rtc['deposit'])[0])
                                                        )
                                                        + str(
                                                            str(
                                                                list(rtc['currency'])[0]
                                                            )
                                                        )
                                                        + ' and security deposit '
                                                        + str(
                                                            rtc[
                                                                'security_deposit'
                                                            ].values[0]
                                                        )
                                                        + ' '
                                                        + str(
                                                            str(
                                                                list(rtc['currency'])[0]
                                                            )
                                                        )
                                                        + '. '
                                                    )
                                                    if (
                                                        rtc['config_tags'].isna().sum()
                                                        == 0
                                                    ):
                                                        prop_configs += (
                                                            'Privacies: '
                                                            + list(rtc['config_tags'])[
                                                                0
                                                            ]
                                                            + '.'
                                                        )

                                                    df_configs = pd.DataFrame(
                                                        [
                                                            [
                                                                str(
                                                                    list(
                                                                        rtc['config_id']
                                                                    )[0]
                                                                )
                                                            ]
                                                        ],
                                                        columns=['prop_config_id'],
                                                    )
                                                    df_configs.to_csv(
                                                        os.path.join(
                                                            univ_folder_path,
                                                            'prop_configs.csv',
                                                        ),
                                                        mode='a',
                                                        header=False,
                                                        index=False,
                                                    )

                                                    if (
                                                        str(list(rtc['config_id'])[0])
                                                        in prop_conf_amenities
                                                    ):
                                                        prop_configs += (
                                                            ' Addtional room amenities: '
                                                            + ', '.join(
                                                                [
                                                                    ame
                                                                    for ame in prop_conf_amenities[
                                                                        str(
                                                                            list(
                                                                                rtc[
                                                                                    'config_id'
                                                                                ]
                                                                            )[0]
                                                                        )
                                                                    ]
                                                                ]
                                                            )
                                                        )
                                                    prop_facility = ''
                                                    proper_id = list(prop_['id'])[0]

                                                    if proper_id in prop_amenities:
                                                        if (
                                                            'COMMUNITY'
                                                            in prop_amenities[proper_id]
                                                        ):
                                                            prop_facility += (
                                                                'Community facilities: '
                                                                + ', '.join(
                                                                    [
                                                                        ame
                                                                        for ame in prop_amenities[
                                                                            proper_id
                                                                        ][
                                                                            'COMMUNITY'
                                                                        ]
                                                                    ]
                                                                )
                                                                + '. '
                                                            )
                                                        if (
                                                            'APARTMENT'
                                                            in prop_amenities[proper_id]
                                                        ):
                                                            prop_facility += (
                                                                'Apartment facilities: '
                                                                + ', '.join(
                                                                    [
                                                                        ame
                                                                        for ame in prop_amenities[
                                                                            proper_id
                                                                        ][
                                                                            'APARTMENT'
                                                                        ]
                                                                    ]
                                                                )
                                                                + '. '
                                                            )

                                                    if (
                                                        prop_['property_tags']
                                                        .isna()
                                                        .sum()
                                                        > 0
                                                    ):
                                                        print(
                                                            'Property:',
                                                            d_prop_name,
                                                            ',',
                                                            d_city,
                                                            ' managed by',
                                                            manager,
                                                            ' is ',
                                                            d_distance_category,
                                                            'accomodation to University: ',
                                                            d_university_name,
                                                            ',',
                                                            d_city,
                                                            'around ',
                                                            d_distance_walking,
                                                            'away . Guarrantor :',
                                                            grntr[guarantor],
                                                            '.Installment plan :',
                                                            plan_[j],
                                                            '. Offers :',
                                                            offer_prop,
                                                            '. ',
                                                            prop_facility,
                                                            prop_configs,
                                                            ' More details and booking at '
                                                            + list(prop['URL'])[0],
                                                        )
                                                    else:
                                                        print(
                                                            'Property:',
                                                            d_prop_name,
                                                            ',',
                                                            d_city,
                                                            ' managed by',
                                                            manager,
                                                            ' is ',
                                                            d_distance_category,
                                                            'accomodation to University: ',
                                                            d_university_name,
                                                            ',',
                                                            d_city,
                                                            'around ',
                                                            d_distance_walking,
                                                            'away . Guarrantor :',
                                                            grntr[guarantor],
                                                            '.Installment plan :',
                                                            plan_[j],
                                                            '. Offers :',
                                                            offer_prop,
                                                            '. Property facilities: ',
                                                            str(
                                                                list(
                                                                    prop_[
                                                                        'property_tags'
                                                                    ]
                                                                )[0]
                                                            ),
                                                            '. ',
                                                            prop_facility,
                                                            prop_configs,
                                                            ' More details and booking at '
                                                            + list(prop['URL'])[0],
                                                        )
                                                    # print("Distance category(",d_distance_category,"property or rooms). University: ",d_university_name,",",d_city,",",d_country,", distance from property",d_prop_name,",",d_city," is ",d_distance_walking,". This property is managed by property manager",manager,manager_enc,". Starting price or cheapest room of this property starts at, ",uni_dist_min_price_prop[d_prop_name],". Guarrantor :",grntr[guarantor],".Installment plan :",plan_[j],". Cashbacks and offers :",offer_prop,". Available for :",sorted_intake_list,".",prop_configs,"Link to the property, URL:"+list(prop['URL'])[0],"\n")

                # Restore the original standard output
                sys.stdout = original_stdout

                # Inform the user that the output has been saved
                print(f'Supply Config Output has been saved to {output_file}')

                ## ethee
                # Specify the file path
                file_path = output_file  # Replace 'your_file.txt' with the path to your text file

                try:
                    # Open the file for reading
                    with open(file_path, 'r') as file:
                        # Read the entire contents of the file
                        file_contents = file.read()

                    # Now, 'file_contents' contains the text from the file
                #     print(file_contents)

                except FileNotFoundError:
                    print(f"The file '{file_path}' was not found.")
                except Exception as e:
                    print(f'An error occurred: {str(e)}')

                raw_text = file_contents
                '''Divide the input data into chunks
                    This will help in reducing the embedding size as we will see in the code
                    as well as reduce the token size for the query,'''
                text_splitter = CharacterTextSplitter(
                    separator='\n',
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len,
                )
                texts = text_splitter.split_text(raw_text)

                embeddings = OpenAIEmbeddings(
                    model='text-embedding-3-small',
                    deployment='text-embedding-3-small',
                    disallowed_special=(),
                )
                docsearch = FAISS.from_texts(texts, embeddings)

                config_pickle = os.path.join(univ_folder_path, 'supply_config.pkl')
                with open(config_pickle, 'wb') as f:
                    pickle.dump(docsearch, f)
                ## ethee

            print(f'Created folder for {city_name} at {city_folder_path}')
