"""
A small, uncontroversial list of well-known legitimate domains, used for two
things:

1. Training augmentation (augment_dataset.py) -- feature_dataset_v2.csv's
   "legitimate" class was sampled almost entirely from deep content links
   (median path length 22 chars) and barely contains bare-root URLs
   (google.com, microsoft.com, ...), while the "scam" class is disproportionately
   bare-root (54% vs 12%). That sampling gap made the trained model flag
   ordinary homepage URLs as scams. These domains, rendered as bare/https/www
   root URLs with status=1, patch that gap with real examples of what a
   legitimate root URL actually looks like.

2. A safety-net allowlist in risk_engine.py, since a few hundred augmented
   rows out of 565k+ training rows can shift but not fully override the
   dataset's statistical pull -- production phishing detectors combine ML
   scores with exactly this kind of reputation/allowlist layer.
"""

KNOWN_LEGITIMATE_DOMAINS = [
    # Search / tech platforms
    "google.com", "bing.com", "duckduckgo.com", "yahoo.com",
    "microsoft.com", "apple.com", "amazon.com", "meta.com", "facebook.com",
    "instagram.com", "twitter.com", "x.com", "linkedin.com", "reddit.com",
    "pinterest.com", "tiktok.com", "snapchat.com", "youtube.com", "netflix.com",
    "spotify.com", "twitch.tv", "discord.com", "slack.com", "zoom.us",
    "dropbox.com", "box.com", "adobe.com", "salesforce.com", "oracle.com",
    "ibm.com", "intel.com", "amd.com", "nvidia.com", "samsung.com", "sony.com",
    "cisco.com", "vmware.com", "sap.com", "shopify.com", "squarespace.com",
    "wordpress.com", "wix.com", "godaddy.com", "namecheap.com", "cloudflare.com",
    "akamai.com", "digitalocean.com", "heroku.com", "vercel.com", "netlify.com",
    # Dev / education
    "github.com", "gitlab.com", "bitbucket.org", "stackoverflow.com",
    "stackexchange.com", "python.org", "nodejs.org", "npmjs.com", "docker.com",
    "kubernetes.io", "wikipedia.org", "wikimedia.org", "khanacademy.org",
    "coursera.org", "udemy.com", "edx.org", "mit.edu", "harvard.edu",
    "stanford.edu", "berkeley.edu", "ox.ac.uk", "cam.ac.uk",
    # News / media
    "bbc.com", "cnn.com", "nytimes.com", "reuters.com", "bloomberg.com",
    "theguardian.com", "washingtonpost.com", "forbes.com", "wsj.com",
    "npr.org", "aljazeera.com", "apnews.com",
    # E-commerce / finance (legitimate domains -- common phishing targets,
    # which is exactly why the model needs real examples of the genuine ones)
    "ebay.com", "etsy.com", "walmart.com", "target.com", "bestbuy.com",
    "paypal.com", "stripe.com", "visa.com", "mastercard.com",
    "chase.com", "bankofamerica.com", "wellsfargo.com", "citibank.com",
    "hsbc.com", "americanexpress.com", "capitalone.com",
    # Government / org
    "usa.gov", "irs.gov", "gov.uk", "europa.eu", "un.org", "who.int",
    "nasa.gov", "nih.gov", "cdc.gov",
    # Travel / misc consumer
    "booking.com", "airbnb.com", "expedia.com", "tripadvisor.com",
    "uber.com", "lyft.com", "doordash.com", "airbnb.co.uk",
    "yelp.com", "imdb.com", "quora.com", "medium.com", "wordpress.org",
    # Jobs / productivity / crypto / real estate (added after testing showed
    # these scoring high despite being clearly legitimate)
    "indeed.com", "glassdoor.com", "monster.com", "ziprecruiter.com",
    "canva.com", "notion.so", "figma.com", "asana.com", "trello.com",
    "coinbase.com", "binance.com", "kraken.com", "zillow.com", "realtor.com",
    "chess.com", "duolingo.com", "grammarly.com", "canva.cn",
]
