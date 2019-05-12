from micromort.newstweets import classify
from micromort.models.trained_models.svm_mean_embeddings import Classifier, MeanEmbeddingVectorizer, PolarityClassifier
from micromort.newstweets.location_fetcher import Location_fetcher
from micromort.newstweets.news_tweet_scraper import News_tweet_scraper

from micromort.data_stores.mongodb import getConnection



def main():


    mongo_db_name = "micromort"
    mongo_collection_name = "newstweets_old"
    mongoClient = getConnection(mongo_db_name, mongo_collection_name)
    l = Location_fetcher()
    n = News_tweet_scraper()
    clf = classify.Classify()



    doc = mongoClient.find_one()
    country, polarity, domain, url = "","","",""
    article = {}
    try:
        location = l.get_location(doc)
        if location is not None:
            country = location[1].__dict__["country"]

        url = n.getUrls(doc)[0]
        
        article = n.scrape(url)

    except Exception as ex:
        print ex
        print doc["id"]
    
    domain = clf.getLifeDomains(article["title"], article["text"])
    polarity = clf.getPolarity(article["title"], article["text"])
    print domain, polarity, country, url
    print "Complete"


if __name__ == "__main__":
    main()
