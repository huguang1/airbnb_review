import time

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split

scaler = StandardScaler()


def bed_bedroom(beds, bedrooms):
    beds_null = beds.isnull()
    bed_rooms_null = bedrooms.isnull()
    for i, v in beds_null.iteritems():
        if v:
            if bed_rooms_null[i]:
                beds[i] = bedrooms[i] = 1
            else:
                beds[i] = bedrooms[i]
    for i, v in bed_rooms_null.iteritems():
        if v:
            bedrooms[i] = beds[i]
    return beds, bedrooms


def price_deal(price):
    for i, v in price.iteritems():
        v = v.strip('$')
        v = v.replace(',', '')
        if float(v) > 2000:
            price[i] = 2000
        else:
            price[i] = float(v)
    return price


def response_rate_deal(host_response_rate):
    response_rate_null = host_response_rate.isnull()
    for i, v in response_rate_null.iteritems():
        if not v:
            a = host_response_rate[i]
            a = a.strip('%')
            host_response_rate[i] = int(a)
        else:
            host_response_rate[i] = 0
    return host_response_rate


def acceptance_rate_deal(host_acceptance_rate):
    acceptance_rate_null = host_acceptance_rate.isnull()
    for i, v in acceptance_rate_null.iteritems():
        if not v:
            a = host_acceptance_rate[i]
            a = a.strip('%')
            host_acceptance_rate[i] = int(a)
        else:
            host_acceptance_rate[i] = 100
    return host_acceptance_rate


def superhost_deal(host_is_superhost):
    for i, v in host_is_superhost.iteritems():
        if v == 't':
            host_is_superhost[i] = 1
        else:
            host_is_superhost[i] = 0
    return host_is_superhost


def availability_deal(has_availability):
    for i, v in has_availability.iteritems():
        if v == 't':
            has_availability[i] = 1
        else:
            has_availability[i] = 0
    return has_availability


def bookable_deal(instant_bookable):
    for i, v in instant_bookable.iteritems():
        if v == 't':
            instant_bookable[i] = 1
        else:
            instant_bookable[i] = 0
    return instant_bookable


def machine():
    listings = pd.read_csv('listings1.csv', encoding='unicode_escape')
    df1 = listings.drop(listings[np.isnan(listings['review_scores_rating'])].index)
    df2 = df1.drop(df1[np.isnan(df1['review_scores_accuracy'])].index)
    df3 = df2.drop(df2[np.isnan(df2['review_scores_cleanliness'])].index)
    df4 = df3.drop(df3[np.isnan(df3['review_scores_checkin'])].index)
    df5 = df4.drop(df4[np.isnan(df4['review_scores_communication'])].index)
    df6 = df5.drop(df5[np.isnan(df5['review_scores_location'])].index)
    listings = df6.drop(df6[np.isnan(df6['review_scores_value'])].index)
    host_response_rate = listings.loc[:, 'host_response_rate']
    host_acceptance_rate = listings.loc[:, 'host_acceptance_rate']
    host_listings_count = listings.loc[:, 'host_listings_count']
    accommodates = listings.loc[:, 'accommodates']
    bedrooms = listings.loc[:, 'bedrooms']
    beds = listings.loc[:, 'beds']
    price = listings.loc[:, 'price']
    minimum_nights = listings.loc[:, 'minimum_nights']
    maximum_nights = listings.loc[:, 'maximum_nights']
    number_of_reviews = listings.loc[:, 'number_of_reviews']
    number_of_reviews_ltm = listings.loc[:, 'number_of_reviews_ltm']
    number_of_reviews_l30d = listings.loc[:, 'number_of_reviews_l30d']
    reviews_per_month = listings.loc[:, 'reviews_per_month']

    # 是否处理
    host_is_superhost = listings.loc[:, 'host_is_superhost']
    has_availability = listings.loc[:, 'has_availability']
    instant_bookable = listings.loc[:, 'instant_bookable']

    # 房子的评分
    review_scores_rating = listings.loc[:, 'review_scores_rating']
    review_scores_accuracy = listings.loc[:, 'review_scores_accuracy']
    review_scores_cleanliness = listings.loc[:, 'review_scores_cleanliness']
    review_scores_checkin = listings.loc[:, 'review_scores_checkin']
    review_scores_communication = listings.loc[:, 'review_scores_communication']
    review_scores_location = listings.loc[:, 'review_scores_location']
    review_scores_value = listings.loc[:, 'review_scores_value']

    listings.loc[listings['neighbourhood_cleansed'].str.contains('Fingal'), 'Fingal'] = 1
    listings.loc[listings['neighbourhood_cleansed'].str.contains('Dublin City'), 'dublin_city'] = 1
    listings.loc[listings['neighbourhood_cleansed'].str.contains('Dn Laoghaire-Rathdown'), 'dn_laoghaire_rathdown'] = 1
    listings.loc[listings['neighbourhood_cleansed'].str.contains('South Dublin'), 'south_dublin'] = 1

    listings.loc[listings['room_type'].str.contains('Private room'), 'private_room'] = 1
    listings.loc[listings['room_type'].str.contains('Entire home/apt'), 'entire_home'] = 1
    listings.loc[listings['room_type'].str.contains('Hotel room'), 'hotel_room'] = 1
    listings.loc[listings['room_type'].str.contains('Shared room'), 'shared_room'] = 1

    listings.loc[listings['amenities'].str.contains('24-hour check-in'), 'check_in_24h'] = 1
    listings.loc[
        listings['amenities'].str.contains('Air conditioning|Central air conditioning'), 'air_conditioning'] = 1
    listings.loc[listings['amenities'].str.contains(
        'Amazon Echo|Apple TV|Game console|Netflix|Projector and screen|Smart TV'), 'high_end_electronics'] = 1
    listings.loc[listings['amenities'].str.contains('BBQ grill|Fire pit|Propane barbeque'), 'bbq'] = 1
    listings.loc[listings['amenities'].str.contains('Balcony|Patio'), 'balcony'] = 1
    listings.loc[listings['amenities'].str.contains(
        'Beach view|Beachfront|Lake access|Mountain view|Ski-in/Ski-out|Waterfront'), 'nature_and_views'] = 1
    listings.loc[listings['amenities'].str.contains('Bed linens'), 'bed_linen'] = 1
    listings.loc[listings['amenities'].str.contains('Breakfast'), 'breakfast'] = 1
    listings.loc[listings['amenities'].str.contains('TV'), 'tv'] = 1
    listings.loc[listings['amenities'].str.contains('Coffee maker|Espresso machine'), 'coffee_machine'] = 1
    listings.loc[listings['amenities'].str.contains('Cooking basics'), 'cooking_basics'] = 1
    listings.loc[listings['amenities'].str.contains('Dishwasher|Dryer|Washer'), 'white_goods'] = 1
    listings.loc[listings['amenities'].str.contains('Elevator'), 'elevator'] = 1
    listings.loc[listings['amenities'].str.contains('Exercise equipment|Gym|gym'), 'gym'] = 1
    listings.loc[listings['amenities'].str.contains('Family/kid friendly|Children|children'), 'child_friendly'] = 1
    listings.loc[listings['amenities'].str.contains('parking'), 'parking'] = 1
    listings.loc[listings['amenities'].str.contains('Garden|Outdoor|Sun loungers|Terrace'), 'outdoor_space'] = 1
    listings.loc[listings['amenities'].str.contains('Host greets you'), 'host_greeting'] = 1
    listings.loc[
        listings['amenities'].str.contains('Hot tub|Jetted tub|hot tub|Sauna|Pool|pool'), 'hot_tub_sauna_or_pool'] = 1
    listings.loc[listings['amenities'].str.contains('Internet|Pocket wifi|Wifi'), 'internet'] = 1
    listings.loc[listings['amenities'].str.contains('Long term stays allowed'), 'long_term_stays'] = 1
    listings.loc[listings['amenities'].str.contains('Pets|pet|Cat(s)|Dog(s)'), 'pets_allowed'] = 1
    listings.loc[listings['amenities'].str.contains('Private entrance'), 'private_entrance'] = 1
    listings.loc[listings['amenities'].str.contains('Safe|Security system'), 'secure'] = 1
    listings.loc[listings['amenities'].str.contains('Self check-in'), 'self_check_in'] = 1
    listings.loc[listings['amenities'].str.contains('Smoking allowed'), 'smoking_allowed'] = 1
    listings.loc[listings['amenities'].str.contains('Step-free access|Wheelchair|Accessible'), 'accessible'] = 1
    listings.loc[listings['amenities'].str.contains('Suitable for events'), 'event_suitable'] = 1

    cols_to_replace_nulls = listings.iloc[:, 63:].columns
    listings[cols_to_replace_nulls] = listings[cols_to_replace_nulls].fillna(0)

    # Fingal = listings.loc[:, 'Fingal']
    # dublin_city = listings.loc[:, 'dublin_city']
    # dn_laoghaire_rathdown = listings.loc[:, 'dn_laoghaire_rathdown']
    # south_dublin = listings.loc[:, 'south_dublin']
    #
    # private_room = listings.loc[:, 'private_room']
    # entire_home = listings.loc[:, 'entire_home']
    # hotel_room = listings.loc[:, 'hotel_room']
    # shared_room = listings.loc[:, 'shared_room']

    check_in_24h = listings.loc[:, 'check_in_24h']
    air_conditioning = listings.loc[:, 'air_conditioning']
    high_end_electronics = listings.loc[:, 'high_end_electronics']
    bbq = listings.loc[:, 'bbq']
    balcony = listings.loc[:, 'balcony']
    nature_and_views = listings.loc[:, 'nature_and_views']
    bed_linen = listings.loc[:, 'bed_linen']
    breakfast = listings.loc[:, 'breakfast']
    tv = listings.loc[:, 'tv']
    coffee_machine = listings.loc[:, 'coffee_machine']
    cooking_basics = listings.loc[:, 'cooking_basics']
    white_goods = listings.loc[:, 'white_goods']
    elevator = listings.loc[:, 'elevator']
    gym = listings.loc[:, 'gym']
    child_friendly = listings.loc[:, 'child_friendly']
    parking = listings.loc[:, 'parking']
    outdoor_space = listings.loc[:, 'outdoor_space']
    host_greeting = listings.loc[:, 'host_greeting']
    hot_tub_sauna_or_pool = listings.loc[:, 'hot_tub_sauna_or_pool']
    internet = listings.loc[:, 'internet']
    long_term_stays = listings.loc[:, 'long_term_stays']
    pets_allowed = listings.loc[:, 'pets_allowed']
    private_entrance = listings.loc[:, 'private_entrance']
    secure = listings.loc[:, 'secure']
    self_check_in = listings.loc[:, 'self_check_in']
    smoking_allowed = listings.loc[:, 'smoking_allowed']
    accessible = listings.loc[:, 'accessible']
    event_suitable = listings.loc[:, 'event_suitable']

    beds, bedrooms = bed_bedroom(beds, bedrooms)
    price = price_deal(price)
    host_response_rate = response_rate_deal(host_response_rate)
    host_is_superhost = superhost_deal(host_is_superhost)
    has_availability = availability_deal(has_availability)
    instant_bookable = bookable_deal(instant_bookable)
    features = np.column_stack((
        host_response_rate,  #
        host_listings_count,  # -0.169
        accommodates,  #
        bedrooms,  #
        beds,  #
        price,  # -0.11
        minimum_nights,  #
        maximum_nights,  #
        number_of_reviews,  #
        number_of_reviews_ltm,  #
        number_of_reviews_l30d,  #
        reviews_per_month,  #
        host_is_superhost,  # 0.135
        has_availability,
        instant_bookable,  #
        # Fingal,
        # dublin_city,  # -0.108
        # dn_laoghaire_rathdown,  # -0.08
        # south_dublin,
        # private_room,
        # entire_home,
        # hotel_room,
        # shared_room,  # -0.08
        check_in_24h,
        air_conditioning,
        high_end_electronics,  # 0.079
        bbq,  # 0.072
        balcony,
        nature_and_views,
        bed_linen,
        breakfast,  # 0.0845
        tv,
        coffee_machine,  # 0.1
        cooking_basics,
        white_goods,
        elevator,
        gym,
        child_friendly,  # 0.93
        parking,  # 0.143
        outdoor_space,  # 0.0953
        host_greeting,  # 0.085
        hot_tub_sauna_or_pool,
        internet,
        long_term_stays,  # 0.099
        pets_allowed,
        private_entrance,
        secure,
        # self_check_in,
        smoking_allowed,
        accessible,
        event_suitable,
    ))

    label_encoder = preprocessing.LabelEncoder()
    review_scores_rating = label_encoder.fit_transform(review_scores_rating)

    standar_scaler = preprocessing.StandardScaler()
    features = standar_scaler.fit_transform(features)
    Xtrain, Xtest, ytrain, ytest = train_test_split(features, review_scores_rating, test_size=0.2)

    def ii_c():
        """
        This function just use Ridge to replace Lasso.
        :return:
        """
        c_range = [0.1, 0.3, 1, 10, 100, 1000, 10000]
        mean_error, std_error = [], []
        for c in c_range:
            model = Ridge(alpha=1 / (2 * c))
            kf = KFold(n_splits=5)
            temp = []
            for train, test in kf.split(features):
                model.fit(features[train], review_scores_rating[train])
                ypred = model.predict(features[test])
                from sklearn.metrics import mean_squared_error
                MSE = mean_squared_error(review_scores_rating[test], ypred)
                print('MSE: ' + str(MSE))
                temp.append(mean_squared_error(review_scores_rating[test], ypred))
            mean_error.append(np.array(temp).mean())
            print(temp)
            std_error.append(np.array(temp).std())
        plt.axes(xscale="log")
        plt.errorbar(c_range, mean_error, yerr=std_error)
        plt.xlabel('C')
        plt.ylabel('Mean square error')
        plt.title("RIDGE")  # title
        plt.xlim((0, 12000))
        plt.show()

    ii_c()




    # model = LinearRegression()
    # model.fit(features, review_scores_rating)
    #
    # print(model.coef_)
    # print(model.intercept_)

    # print(model.score(features, review_scores_communication))
    # # a = features[:, 0]
    # pccs = np.corrcoef([i for i in features[:, 0]], review_scores_communication)
    # print(pccs)

    C = 1
    alpha = 1 / (2 * C)
    model = Ridge(alpha)
    model.fit(Xtrain, ytrain)
    # weights = [abs(i) for i in model.coef_]
    weights = model.coef_
    intercept = model.intercept_
    y19_pred = model.predict(Xtest)
    mse_score = mean_squared_error(ytest, y19_pred)
    r2_score = metrics.r2_score(ytest, y19_pred)
    print(mse_score)
    print(r2_score)
    drop_df = pd.DataFrame()
    X19_cols = [
                # "host_id",  # 0.128
                "host_response_rate",  #
                # "host_acceptance_rate",  #
                "host_listings_count",  # -0.169
                "accommodates",  #
                "bedrooms",  #
                "beds",  #
                "price",  # -0.11
                "minimum_nights",  #
                "maximum_nights",  #
                "number_of_reviews",  #
                "number_of_reviews_ltm",  #
                "number_of_reviews_l30d",  #
                "reviews_per_month",  #
                "host_is_superhost",  # 0.135
                "has_availability",
                "instant_bookable",  #
                # "Fingal",
                # "dublin_city",  # -0.108
                # "dn_laoghaire_rathdown",  # -0.08
                # "south_dublin",
                # "private_room",
                # "entire_home",
                # "hotel_room",
                # "shared_room",  # -0.08
                "check_in_24h",
                "air_conditioning",
                "high_end_electronics",  # 0.079
                "bbq",  # 0.072
                "balcony",
                "nature_and_views",
                "bed_linen",
                "breakfast",  # 0.0845
                "tv",
                "coffee_machine",  # 0.1
                "cooking_basics",
                "white_goods",
                "elevator",
                "gym",
                "child_friendly",  # 0.93
                "parking",  # 0.143
                "outdoor_space",  # 0.0953
                "host_greeting",  # 0.085
                "hot_tub_sauna_or_pool",
                "internet",
                "long_term_stays",  # 0.099
                "pets_allowed",
                "private_entrance",
                "secure",
                # "self_check_in",
                "smoking_allowed",
                "accessible",
                "event_suitable", ]

    # df19_cols = listings.columns.tolist()

    drop_df["features"] = X19_cols
    drop_df["weights"] = weights

    num_features = 10
    drop_df.sort_values('weights', ascending=False, inplace=True)
    drop_df.features = drop_df.features[0:num_features]
    drop_df.weights = drop_df.weights[0:num_features]
    # drop_df.features = drop_df.features[-1*num_features:]
    # drop_df.weights = drop_df.weights[-1*num_features:]
    # drop_df.features = drop_df.features
    # drop_df.weights = drop_df.weights
    print(weights)

    fig = plt.figure()

    ax = fig.add_subplot()
    ax.barh(drop_df.features[0:num_features], drop_df.weights[0:num_features])
    # ax.barh(drop_df.features[-1*num_features:], drop_df.weights[-1*num_features:])
    ax.set_title("10 Features for checkin")
    ax.set_xlabel("Weight")

    plt.tight_layout()


if __name__ == '__main__':
    machine()
