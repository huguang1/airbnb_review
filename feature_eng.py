import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas import Series

np.set_printoptions(threshold=np.inf)
from sklearn import linear_model, preprocessing
from numpy import mean

# read dataset
df1 = pd.read_csv("listings.csv", encoding='unicode_escape')
df0 = df1.drop(df1[np.isnan(df1['review_scores_rating'])].index)
# df2 = df1.drop(df1[np.isnan(df1['review_scores_accuracy'])].index)
# df3 = df2.drop(df2[np.isnan(df2['review_scores_cleanliness'])].index)
# df4 = df3.drop(df3[np.isnan(df3['review_scores_checkin'])].index)
# df5 = df4.drop(df4[np.isnan(df4['review_scores_communication'])].index)
# df6 = df5.drop(df5[np.isnan(df5['review_scores_location'])].index)
# df1 = df6.drop(df6[np.isnan(df6['review_scores_value'])].index)

# read features
id = df1.iloc[:, 0]


# process souce features, let city scrape = 0, previuos scrape = 1
def process_source(df1):
    data = df1["source"]
    new_source = []
    for i in data:
        if (i == 'city scrape'):
            new_source.append(0)
        else:
            new_source.append(1)
    return new_source


# process host_responce_time features
def process_host_response_time(df1):
    data = df1["host_response_time"]
    response_time_set = set()
    new_response_time = []
    for i in data:
        response_time_set.add(i)
    for i in data:
        if (i == "a few days or more"):
            new_response_time.append(0)
        elif (i == "within an hour"):
            new_response_time.append(1)
        elif (i == "within a few hours"):
            new_response_time.append(2)
        elif (i == "within a day"):
            new_response_time.append(3)
        else:
            new_response_time.append(0)
    return new_response_time


# process host_responce_rate features and host_accptance_rate features
def process_host_resAndaccp(feature):
    feature = feature.str.strip("%").astype(float) / 100
    mean = np.mean(feature)
    mean = mean.round(decimals=2)
    data_new = []
    for i in feature:
        if np.isnan(i):
            data_new.append(mean)
        else:
            data_new.append(i)
    return data_new


# process features that consist of t and f
def procee_TorF_feature(feature):
    data = []
    for i in feature:
        if i == 't':
            data.append(1)
        elif i == 'f':
            data.append(0)
    return data


# process neighbourhood_cleansed feature
def process_neighbourhood_cleansed(feature):
    data_set = set()
    data = []
    for i in feature:
        data_set.add(i)
    for i in feature:
        if i == "Dn Laoghaire-Rathdown":
            data.append(0)
        elif i == "South Dublin":
            data.append(1)
        elif i == "Dublin City":
            data.append(2)
        elif i == "Fingal":
            data.append(3)
    return data


# process property_type feature
def process_room_type(feature):
    data_set = set()
    data = []
    for i in feature:
        data_set.add(i)
    for i in feature:
        if i == "Entire home/apt":
            data.append(0)
        elif i == "Private room":
            data.append(1)
        elif i == "Shared room":
            data.append(2)
        elif i == "Hotel room":
            data.append(3)
    return data


# process bathrooms_txt feature
def process_bathroom_txt(feature):
    share = []
    no_share = []
    for i in feature:
        i = str(i)

        if (i.find("shared") != -1 or i.find("Shared") != -1):

            if (i.find("half-bath") != -1 or i.find("Half-bath") != -1):
                share.append(0.5)
            else:
                i = i.split(" ")
                share.append(float(i[0]))
        else:
            if (i.find("half-bath") != -1 or i.find("Half-bath") != -1):
                no_share.append(0.5)
            else:
                i = i.split(" ")
                no_share.append(float(i[0]))
    data = np.hstack((share, no_share))
    return data


def process_amenities():
    df1.loc[df1['amenities'].str.contains('24-hour check-in'), 'check_in_24h'] = 1
    df1.loc[df1['amenities'].str.contains('Air conditioning|Central air conditioning'), 'air_conditioning'] = 1
    df1.loc[df1['amenities'].str.contains(
        'Amazon Echo|Apple TV|Game console|Netflix|Projector and screen|Smart TV'), 'high_end_electronics'] = 1
    df1.loc[df1['amenities'].str.contains('BBQ grill|Fire pit|Propane barbeque'), 'bbq'] = 1
    df1.loc[df1['amenities'].str.contains('Balcony|Patio'), 'balcony'] = 1
    df1.loc[df1['amenities'].str.contains(
        'Beach view|Beachfront|Lake access|Mountain view|Ski-in/Ski-out|Waterfront'), 'nature_and_views'] = 1
    df1.loc[df1['amenities'].str.contains('Bed linens'), 'bed_linen'] = 1
    df1.loc[df1['amenities'].str.contains('Breakfast'), 'breakfast'] = 1
    df1.loc[df1['amenities'].str.contains('TV'), 'tv'] = 1
    df1.loc[df1['amenities'].str.contains('Coffee maker|Espresso machine'), 'coffee_machine'] = 1
    df1.loc[df1['amenities'].str.contains('Cooking basics'), 'cooking_basics'] = 1
    df1.loc[df1['amenities'].str.contains('Dishwasher|Dryer|Washer'), 'white_goods'] = 1
    df1.loc[df1['amenities'].str.contains('Elevator'), 'elevator'] = 1
    df1.loc[df1['amenities'].str.contains('Exercise equipment|Gym|gym'), 'gym'] = 1
    df1.loc[df1['amenities'].str.contains('Family/kid friendly|Children|children'), 'child_friendly'] = 1
    df1.loc[df1['amenities'].str.contains('parking'), 'parking'] = 1
    df1.loc[df1['amenities'].str.contains('Garden|Outdoor|Sun loungers|Terrace'), 'outdoor_space'] = 1
    df1.loc[df1['amenities'].str.contains('Host greets you'), 'host_greeting'] = 1
    df1.loc[df1['amenities'].str.contains('Hot tub|Jetted tub|hot tub|Sauna|Pool|pool'), 'hot_tub_sauna_or_pool'] = 1
    df1.loc[df1['amenities'].str.contains('Internet|Pocket wifi|Wifi'), 'internet'] = 1
    df1.loc[df1['amenities'].str.contains('Long term stays allowed'), 'long_term_stays'] = 1
    df1.loc[df1['amenities'].str.contains('Pets|pet|Cat(s)|Dog(s)'), 'pets_allowed'] = 1
    df1.loc[df1['amenities'].str.contains('Private entrance'), 'private_entrance'] = 1
    df1.loc[df1['amenities'].str.contains('Safe|Security system'), 'secure'] = 1
    df1.loc[df1['amenities'].str.contains('Self check-in'), 'self_check_in'] = 1
    df1.loc[df1['amenities'].str.contains('Smoking allowed'), 'smoking_allowed'] = 1
    df1.loc[df1['amenities'].str.contains('Step-free access|Wheelchair|Accessible'), 'accessible'] = 1
    df1.loc[df1['amenities'].str.contains('Suitable for events'), 'event_suitable'] = 1

    return df1[
        ['check_in_24h', 'air_conditioning', 'high_end_electronics', 'nature_and_views', 'bed_linen', 'breakfast', 'tv',
         'coffee_machine',
         'cooking_basics', 'white_goods', 'elevator', 'gym', 'child_friendly', 'parking', 'outdoor_space',
         'host_greeting', 'hot_tub_sauna_or_pool',
         'internet', 'long_term_stays', 'pets_allowed', 'private_entrance', 'secure', 'self_check_in',
         'smoking_allowed', 'accessible', 'event_suitable']]


def process_price(feature):
    data = []
    for i in feature:
        i = i.strip("$")
        i = float(i.replace(",", ""))
        data.append(i)
    return data


##############################################################
# for null values in all features
# n_nan = []
# nan_cols = []
# n_cols = len(df1.columns.tolist())
#
# for col in df1.columns.tolist():
#     if df1[col].isna().sum() > 0:
#         nan_cols.append(col)
#         n_nan.append(df1[col].isna().sum())
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(n_nan)
# ax.set_title("NaN Values in listing data")
# ax.set_ylabel("Number of NaN Values")
# ax.set_xticks(list(range(0, len(n_nan))))
# ax.set_xticklabels(nan_cols, rotation=90)
# ax.grid("on")
# plt.tight_layout()
# plt.show()
# score features
review_scores_rating = df1["review_scores_rating"]
review_scores_accuracy = df1["review_scores_accuracy"]
review_scores_cleanliness = df1["review_scores_cleanliness"]
review_scores_checkin = df1["review_scores_checkin"]
review_scores_communication = df1["review_scores_communication"]
review_scores_location = df1["review_scores_location"]
review_scores_value = df1["review_scores_value"]
# process features
source = process_source(df1)
host_response_time = process_host_response_time(df1)
host_response_rate = process_host_resAndaccp(df1["host_response_rate"])
host_accptance_rate = process_host_resAndaccp(df1["host_acceptance_rate"])
host_is_superhost = procee_TorF_feature(df1["host_is_superhost"])
host_has_profile_pic = procee_TorF_feature(df1["host_has_profile_pic"])
host_identity_verified = procee_TorF_feature(df1["host_identity_verified"])
neighbourhood_cleansed = process_neighbourhood_cleansed(df1["neighbourhood_cleansed"])
room_type = process_room_type(df1["room_type"])
accommodates = df1["accommodates"]
bathrooms_text = process_bathroom_txt(df1["bathrooms_text"])
bedrooms = df1["bedrooms"]
beds = df1["beds"]
amenities = process_amenities().fillna(0)
price = process_price(df1["price"])
minimum_nights_avg_ntm = df1["minimum_nights_avg_ntm"]
maximum_nights_avg_ntm = df1["maximum_nights_avg_ntm"]
number_of_reviews = df1["number_of_reviews"]
instant_bookable = procee_TorF_feature(df1["instant_bookable"])
reviews_per_month = df1["reviews_per_month"]
availability_90 = df1["availability_90"]

frames = [Series(source), Series(host_is_superhost), Series(host_response_rate), Series(host_response_time),
          Series(host_accptance_rate), Series(host_has_profile_pic), Series(host_identity_verified),
          Series(neighbourhood_cleansed), Series(room_type), Series(accommodates), Series(bathrooms_text),
          Series(bedrooms), Series(beds), Series(price), Series(minimum_nights_avg_ntm), Series(maximum_nights_avg_ntm),
          Series(number_of_reviews), Series(instant_bookable), Series(reviews_per_month), Series(availability_90)]
new = pd.concat(frames, axis=1)
new = pd.DataFrame(new)
new.columns = ['source', 'host_is_superhost', 'host_response_rate', 'host_response_time', 'host_accptance_rate',
               'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed', 'room_type',
               'accommodates', 'bathrooms_text', 'bedrooms', 'beds', 'price', 'minimum_nights_avg_ntm',
               'maximum_nights_avg_ntm', 'number_of_reviews', 'instant_bookable', 'reviews_per_month',
               'availability_90']
frames2 = [new, amenities]
df3 = pd.concat(frames2, axis=1)

df3['bathrooms_text'] = df3['bathrooms_text'].fillna(0)
df3['bedrooms'] = df3['bedrooms'].fillna(1)
df3['beds'] = df3['beds'].fillna(1)
df3['availability_90'] = df3['availability_90'].fillna(0)
df3['reviews_per_month'] = df3['reviews_per_month'].fillna(mean(df3['reviews_per_month']))

model = linear_model.LinearRegression()
model.fit(df3, review_scores_checkin.fillna(mean(review_scores_checkin)))
weights = model.coef_
df_diagram = pd.DataFrame(weights, columns=['weight'], index=df3.columns)
df_diagram.sort_values('weight', ascending=False, inplace=True)
# print(df_diagram.head(10))
# num_features = 10

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.barh(df_diagram.index[0:10], df_diagram.weight[0:10])
# # ax.barh(df_diagram.index, df_diagram.weight)
# plt.title("10 Features for checkin")
# plt.xlabel("Weight")
# plt.show()

fig = plt.figure()

ax = fig.add_subplot()
# ax.barh(drop_df.features[0:num_features], drop_df.weights[0:num_features])
ax.barh(df_diagram.index[0:10], df_diagram.weight[0:10])
ax.set_title("10 Features for checkin")
ax.set_xlabel("Weight")

plt.tight_layout()

# df_train = df3[["event_suitable", 'pets_allowed', 'host_is_superhost', 'internet', 'host_greeting', 'self_check_in',
#                 'child_friendly', 'nature_and_views', 'coffee_machine', 'outdoor_space']]
# model.fit(df_train, review_scores_value.fillna(mean(review_scores_value)))
# score = model.score(df_train, review_scores_value.fillna(mean(review_scores_value)))
# print(score)

# # store features
# listings = {"id": id,
#             "source": source,
#             "host_response_time": host_response_time,
#             "host_response_rate": host_response_rate,
#             "host_accptance_rate": host_accptance_rate,
#             "host_is_superhost": host_is_superhost,
#             "host_has_profile_pic": host_has_profile_pic,
#             "host_identity_verified": host_identity_verified,
#             "neighbourhood_cleansed": neighbourhood_cleansed,
#             "room_type": room_type,
#             "accommodates": accommodates,
#             "bathrooms_text": bathrooms_text,
#             "bedrooms": bedrooms,
#             "beds": beds,
#             "amenities": amenities,
#             "price": price,
#             "minimum_nights_avg_ntm": minimum_nights_avg_ntm,
#             "maximum_nights_avg_ntm": maximum_nights_avg_ntm,
#             "number_of_reviews": number_of_reviews,
#             "instant_bookable": instant_bookable,
#             "reviews_per_month": reviews_per_month
#             }
