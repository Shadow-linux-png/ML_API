import pickle
import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
import logging
import random
from faker import Faker
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configuration
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'metadata.json')

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize Faker for generating realistic text
fake = Faker()

def generate_sentiment_dataset(num_samples=1000):
    """Generate a comprehensive sentiment dataset using Faker and templates."""
    
    # Positive templates and components
    positive_templates = [
        "I {love} this {product}!",
        "This {product} is {amazing}!",
        "{excellent} {quality} and {fast} {shipping}",
        "{highly} {recommend} this {product}",
        "{outstanding} {performance} and {features}",
        "{fantastic} {experience} {overall}",
        "{top} {notch} {quality} and {design}",
        "{exceeded} my {expectations} {completely}",
        "{professional} and {reliable} {service}",
        "{innovative} and {well}-built {product}",
        "{absolutely} {wonderful} {purchase}",
        "{best} {investment} I've {made}",
        "{impressive} {build} {quality}",
        "{customer} {service} was {exceptional}",
        "{product} {works} {exactly} as {described}",
        "{great} {value} for {money}",
        "{would} {definitely} {buy} {again}",
        "{perfect} for my {needs}",
        "{above} and {beyond} my {expectations}",
        "{couldn't} be {happier} with this {purchase}",
        "{five} {stars} all the {way}",
        "{excellent} {customer} {support}",
        "{high} {quality} {materials}",
        "{well} {worth} the {price}",
        "{love} the {design} and {functionality}"
    ]
    
    positive_words = {
        'love': ['love', 'adore', 'cherish', 'enjoy'],
        'product': ['product', 'item', 'purchase', 'buy', 'investment'],
        'amazing': ['amazing', 'fantastic', 'wonderful', 'excellent', 'outstanding'],
        'excellent': ['excellent', 'superb', 'magnificent', 'exceptional', 'terrific'],
        'quality': ['quality', 'craftsmanship', 'build', 'construction'],
        'fast': ['fast', 'quick', 'rapid', 'speedy', 'prompt'],
        'shipping': ['shipping', 'delivery', 'dispatch', 'service'],
        'highly': ['highly', 'strongly', 'definitely', 'absolutely'],
        'recommend': ['recommend', 'suggest', 'advise', 'endorse'],
        'outstanding': ['outstanding', 'excellent', 'superb', 'exceptional'],
        'performance': ['performance', 'functionality', 'operation', 'execution'],
        'features': ['features', 'capabilities', 'functions', 'attributes'],
        'fantastic': ['fantastic', 'wonderful', 'amazing', 'great', 'excellent'],
        'experience': ['experience', 'journey', 'process', 'interaction'],
        'overall': ['overall', 'altogether', 'in total', 'completely'],
        'top': ['top', 'premium', 'first-class', 'superior'],
        'notch': ['notch', 'quality', 'grade', 'level'],
        'design': ['design', 'style', 'appearance', 'aesthetics'],
        'exceeded': ['exceeded', 'surpassed', 'went beyond', 'transcended'],
        'expectations': ['expectations', 'hopes', 'anticipations', 'requirements'],
        'professional': ['professional', 'expert', 'skilled', 'competent'],
        'reliable': ['reliable', 'dependable', 'trustworthy', 'consistent'],
        'service': ['service', 'support', 'assistance', 'help'],
        'innovative': ['innovative', 'creative', 'original', 'inventive'],
        'well': ['well', 'expertly', 'skillfully', 'professionally'],
        'built': ['built', 'constructed', 'crafted', 'manufactured'],
        'absolutely': ['absolutely', 'completely', 'totally', 'entirely'],
        'wonderful': ['wonderful', 'marvelous', 'splendid', 'superb'],
        'purchase': ['purchase', 'buy', 'acquisition', 'investment'],
        'best': ['best', 'greatest', 'finest', 'superior'],
        'investment': ['investment', 'purchase', 'acquisition', 'buy'],
        'made': ['made', 'done', 'completed', 'executed'],
        'impressive': ['impressive', 'remarkable', 'extraordinary', 'notable'],
        'build': ['build', 'construction', 'craftsmanship', 'quality'],
        'customer': ['customer', 'client', 'user', 'buyer'],
        'works': ['works', 'functions', 'operates', 'performs'],
        'exactly': ['exactly', 'precisely', 'accurately', 'perfectly'],
        'described': ['described', 'advertised', 'presented', 'shown'],
        'great': ['great', 'excellent', 'wonderful', 'fantastic'],
        'value': ['value', 'worth', 'merit', 'quality'],
        'money': ['money', 'price', 'cost', 'investment'],
        'would': ['would', 'will', 'shall', 'definitely'],
        'definitely': ['definitely', 'certainly', 'absolutely', 'surely'],
        'buy': ['buy', 'purchase', 'acquire', 'invest'],
        'again': ['again', 'repeatedly', 'continuously', 'regularly'],
        'perfect': ['perfect', 'ideal', 'flawless', 'excellent'],
        'needs': ['needs', 'requirements', 'demands', 'expectations'],
        'above': ['above', 'beyond', 'surpassing', 'exceeding'],
        'beyond': ['beyond', 'above', 'past', 'outside'],
        'happier': ['happier', 'more satisfied', 'more pleased', 'more content'],
        'five': ['five', '5', 'excellent', 'perfect'],
        'stars': ['stars', 'rating', 'score', 'points'],
        'way': ['way', 'manner', 'method', 'approach'],
        'support': ['support', 'help', 'assistance', 'service'],
        'high': ['high', 'premium', 'superior', 'excellent'],
        'materials': ['materials', 'components', 'parts', 'elements'],
        'worth': ['worth', 'valuable', 'precious', 'cost-effective'],
        'price': ['price', 'cost', 'value', 'investment'],
        'functionality': ['functionality', 'features', 'capabilities', 'performance']
    }
    
    # Negative templates and components
    negative_templates = [
        "This {product} is {terrible}!",
        "I {hate} this {product}",
        "{poor} {customer} {service}, very {disappointed}",
        "{worst} {purchase} ever {made}",
        "{complete} {waste} of {time} and {money}",
        "{defective} {product}, {doesn't} {work}",
        "{regret} {buying} this {immediately}",
        "{never} {ordering} from here {again}",
        "{scam}, {don't} {trust} this {seller}",
        "{useless} and {overpriced} {junk}",
        "{cheap} and {poorly} {made}",
        "{product} {broke} after one {use}",
        "{misleading} {advertisement}",
        "{terrible} {user} {experience}",
        "{would} not {recommend} to {anyone}",
        "{product} {arrived} {damaged}",
        "{disappointing} {quality} and {performance}",
        "{not} {worth} the {money}",
        "{customer} {service} was {unhelpful}",
        "{product} {failed} to {meet} {expectations}",
        "{would} {give} zero {stars} if {possible}",
        "{avoid} this {product} at all {costs}",
        "{poor} {value} for {money}",
        "{cheap} {materials} and {construction}",
        "{doesn't} {work} as {advertised}",
        "{total} {disappointment} and {waste}"
    ]
    
    negative_words = {
        'terrible': ['terrible', 'awful', 'horrible', 'dreadful', 'appalling'],
        'hate': ['hate', 'despise', 'loathe', 'detest', 'abhor'],
        'poor': ['poor', 'bad', 'inadequate', 'substandard', 'inferior'],
        'customer': ['customer', 'client', 'user', 'buyer'],
        'service': ['service', 'support', 'assistance', 'help'],
        'disappointed': ['disappointed', 'let down', 'unsatisfied', 'dissatisfied'],
        'worst': ['worst', 'poorest', 'most terrible', 'most awful'],
        'purchase': ['purchase', 'buy', 'acquisition', 'investment'],
        'made': ['made', 'done', 'completed', 'executed'],
        'complete': ['complete', 'total', 'absolute', 'utter'],
        'waste': ['waste', 'squander', 'misuse', 'fritter'],
        'time': ['time', 'effort', 'energy', 'resources'],
        'money': ['money', 'cash', 'funds', 'investment'],
        'defective': ['defective', 'faulty', 'broken', 'malfunctioning'],
        'product': ['product', 'item', 'device', 'equipment'],
        "doesn't": ["doesn't", "does not", "fails to", "won't"],
        'work': ['work', 'function', 'operate', 'perform'],
        'regret': ['regret', 'rue', 'lament', 'be sorry about'],
        'buying': ['buying', 'purchasing', 'acquiring', 'getting'],
        'immediately': ['immediately', 'right away', 'at once', 'instantly'],
        'never': ['never', 'not ever', 'under no circumstances', 'absolutely not'],
        'ordering': ['ordering', 'buying', 'purchasing', 'shopping'],
        'again': ['again', 'repeatedly', 'ever', 'in future'],
        'scam': ['scam', 'fraud', 'trick', 'deception'],
        "don't": ["don't", "do not", "never", "avoid"],
        'trust': ['trust', 'believe', 'rely on', 'depend on'],
        'seller': ['seller', 'vendor', 'merchant', 'retailer'],
        'useless': ['useless', 'worthless', 'pointless', 'ineffective'],
        'overpriced': ['overpriced', 'expensive', 'costly', 'pricey'],
        'junk': ['junk', 'garbage', 'trash', 'rubbish'],
        'cheap': ['cheap', 'low-quality', 'poorly made', 'shoddy'],
        'poorly': ['poorly', 'badly', 'inadequately', 'incompetently'],
        'made': ['made', 'constructed', 'built', 'manufactured'],
        'broke': ['broke', 'broken', 'damaged', 'failed'],
        'use': ['use', 'usage', 'operation', 'application'],
        'misleading': ['misleading', 'deceptive', 'false', 'dishonest'],
        'advertisement': ['advertisement', 'ad', 'promotion', 'marketing'],
        'terrible': ['terrible', 'awful', 'horrible', 'dreadful'],
        'user': ['user', 'customer', 'client', 'buyer'],
        'experience': ['experience', 'interaction', 'journey', 'process'],
        'would': ['would', 'will', 'shall', 'definitely'],
        'not': ['not', 'never', 'absolutely not', 'under no circumstances'],
        'recommend': ['recommend', 'suggest', 'advise', 'endorse'],
        'anyone': ['anyone', 'anybody', 'no one', 'nobody'],
        'arrived': ['arrived', 'came', 'was delivered', 'was received'],
        'damaged': ['damaged', 'broken', 'harmed', 'impaired'],
        'disappointing': ['disappointing', 'unsatisfactory', 'inadequate', 'subpar'],
        'quality': ['quality', 'standard', 'level', 'grade'],
        'performance': ['performance', 'functionality', 'operation', 'execution'],
        'worth': ['worth', 'valuable', 'cost-effective', 'worthwhile'],
        'unhelpful': ['unhelpful', 'useless', 'ineffective', 'unproductive'],
        'failed': ['failed', 'didn\'t work', 'was unsuccessful', 'fell short'],
        'meet': ['meet', 'satisfy', 'fulfill', 'achieve'],
        'expectations': ['expectations', 'requirements', 'needs', 'demands'],
        'give': ['give', 'provide', 'offer', 'grant'],
        'zero': ['zero', 'no', 'none', 'negative'],
        'stars': ['stars', 'rating', 'score', 'points'],
        'possible': ['possible', 'conceivable', 'imaginable', 'feasible'],
        'avoid': ['avoid', 'steer clear of', 'stay away from', 'shun'],
        'costs': ['costs', 'expense', 'price', 'sacrifice'],
        'value': ['value', 'worth', 'merit', 'quality'],
        'materials': ['materials', 'components', 'parts', 'elements'],
        'construction': ['construction', 'build', 'assembly', 'manufacturing'],
        "doesn't": ["doesn't", "does not", "fails to", "won't"],
        'advertised': ['advertised', 'promoted', 'marketed', 'described'],
        'total': ['total', 'complete', 'absolute', 'utter'],
        'disappointment': ['disappointment', 'letdown', 'dissatisfaction', 'regret']
    }
    
    # Neutral templates and components
    neutral_templates = [
        "The {product} is {okay}, nothing {special}",
        "{average} {quality} for the {price}",
        "It {works} but has some {issues}",
        "{mixed} {feelings} about this {purchase}",
        "{not} {bad} but could be {better}",
        "{decent} {value} but not {exceptional}",
        "The {product} {meets} basic {expectations}",
        "{standard} {quality} and {performance}",
        "{acceptable} for the {cost}",
        "{neither} {good} nor {bad}",
        "{typical} {product} in this {category}",
        "{reasonable} {price} for what you {get}",
        "{functions} as {described} but {unremarkable}",
        "{adequate} for {basic} {needs}",
        "{ordinary} {design} and {features}",
        "{satisfactory} {performance} overall",
        "{meets} {minimum} {requirements}",
        "{fair} {value} for the {money}",
        "{standard} {customer} {service}",
        "{nothing} {outstanding} but {functional}",
        "{average} {user} {experience}",
        "{basic} {product} with {limited} {features}",
        "{reasonable} {quality} for the {price}",
        "{works} {as} {expected}",
        "{typical} {performance} for this {type}",
        "{adequate} but not {impressive}",
        "{functional} but {lacks} {innovation}",
        "{meets} {specifications} but {unexciting}"
    ]
    
    neutral_words = {
        'product': ['product', 'item', 'device', 'equipment', 'purchase'],
        'okay': ['okay', 'alright', 'fine', 'acceptable', 'satisfactory'],
        'special': ['special', 'exceptional', 'outstanding', 'remarkable'],
        'average': ['average', 'typical', 'standard', 'ordinary', 'normal'],
        'quality': ['quality', 'standard', 'level', 'grade', 'caliber'],
        'price': ['price', 'cost', 'value', 'expense', 'investment'],
        'works': ['works', 'functions', 'operates', 'performs', 'runs'],
        'issues': ['issues', 'problems', 'concerns', 'challenges', 'difficulties'],
        'mixed': ['mixed', 'conflicting', 'varied', 'diverse', 'mixed'],
        'feelings': ['feelings', 'opinions', 'thoughts', 'views', 'impressions'],
        'purchase': ['purchase', 'buy', 'acquisition', 'investment'],
        'bad': ['bad', 'poor', 'inadequate', 'substandard', 'unacceptable'],
        'better': ['better', 'improved', 'enhanced', 'superior', 'upgraded'],
        'decent': ['decent', 'acceptable', 'satisfactory', 'reasonable', 'adequate'],
        'value': ['value', 'worth', 'merit', 'quality', 'benefit'],
        'exceptional': ['exceptional', 'outstanding', 'remarkable', 'extraordinary', 'special'],
        'meets': ['meets', 'satisfies', 'fulfills', 'achieves', 'reaches'],
        'basic': ['basic', 'fundamental', 'essential', 'minimum', 'standard'],
        'expectations': ['expectations', 'requirements', 'needs', 'demands', 'specifications'],
        'standard': ['standard', 'typical', 'ordinary', 'normal', 'regular'],
        'performance': ['performance', 'functionality', 'operation', 'execution', 'results'],
        'acceptable': ['acceptable', 'satisfactory', 'adequate', 'reasonable', 'decent'],
        'cost': ['cost', 'price', 'expense', 'value', 'investment'],
        'neither': ['neither', 'not either', 'none of', 'neither one'],
        'good': ['good', 'excellent', 'great', 'positive', 'favorable'],
        'category': ['category', 'class', 'type', 'group', 'segment'],
        'reasonable': ['reasonable', 'fair', 'justifiable', 'logical', 'sensible'],
        'get': ['get', 'receive', 'obtain', 'acquire', 'gain'],
        'functions': ['functions', 'works', 'operates', 'performs', 'runs'],
        'described': ['described', 'advertised', 'presented', 'shown', 'explained'],
        'unremarkable': ['unremarkable', 'ordinary', 'typical', 'standard', 'average'],
        'adequate': ['adequate', 'sufficient', 'acceptable', 'satisfactory', 'decent'],
        'needs': ['needs', 'requirements', 'demands', 'expectations', 'necessities'],
        'ordinary': ['ordinary', 'typical', 'standard', 'normal', 'regular'],
        'design': ['design', 'style', 'appearance', 'aesthetics', 'look'],
        'features': ['features', 'capabilities', 'functions', 'attributes', 'properties'],
        'satisfactory': ['satisfactory', 'acceptable', 'adequate', 'decent', 'reasonable'],
        'overall': ['overall', 'generally', 'in total', 'altogether', 'completely'],
        'minimum': ['minimum', 'lowest', 'least', 'basic', 'fundamental'],
        'fair': ['fair', 'reasonable', 'just', 'equitable', 'balanced'],
        'money': ['money', 'price', 'cost', 'value', 'investment'],
        'customer': ['customer', 'client', 'user', 'buyer'],
        'service': ['service', 'support', 'assistance', 'help'],
        'nothing': ['nothing', 'none', 'zero', 'lack', 'absence'],
        'outstanding': ['outstanding', 'excellent', 'exceptional', 'remarkable', 'superb'],
        'functional': ['functional', 'working', 'operational', 'practical', 'useful'],
        'user': ['user', 'customer', 'client', 'buyer'],
        'experience': ['experience', 'interaction', 'journey', 'process'],
        'limited': ['limited', 'restricted', 'basic', 'minimal', 'simple'],
        'typical': ['typical', 'standard', 'ordinary', 'normal', 'regular'],
        'type': ['type', 'category', 'class', 'kind', 'sort'],
        'impressive': ['impressive', 'remarkable', 'extraordinary', 'outstanding', 'exceptional'],
        'lacks': ['lacks', 'missing', 'without', 'lacking', 'deficient'],
        'innovation': ['innovation', 'creativity', 'originality', 'novelty', 'ingenuity'],
        'specifications': ['specifications', 'specs', 'requirements', 'details', 'parameters'],
        'unexciting': ['unexciting', 'boring', 'dull', 'uninteresting', 'mundane']
    }
    
    # Generate dataset
    texts = []
    labels = []
    
    # Generate positive samples (40% of dataset)
    positive_count = int(num_samples * 0.4)
    for i in range(positive_count):
        template = random.choice(positive_templates)
        text = template
        
        # Replace placeholders with random variations
        for placeholder in positive_words:
            if f'{{{placeholder}}}' in text:
                replacement = random.choice(positive_words[placeholder])
                text = text.replace(f'{{{placeholder}}}', replacement, 1)
        
        # Add some random variations
        if random.random() < 0.3:
            text += f" {fake.sentence()[:50]}"
        
        texts.append(text)
        labels.append(1)
    
    # Generate negative samples (40% of dataset)
    negative_count = int(num_samples * 0.4)
    for i in range(negative_count):
        template = random.choice(negative_templates)
        text = template
        
        # Replace placeholders with random variations
        for placeholder in negative_words:
            if f'{{{placeholder}}}' in text:
                replacement = random.choice(negative_words[placeholder])
                text = text.replace(f'{{{placeholder}}}', replacement, 1)
        
        # Add some random variations
        if random.random() < 0.3:
            text += f" {fake.sentence()[:50]}"
        
        texts.append(text)
        labels.append(0)
    
    # Generate neutral samples (20% of dataset)
    neutral_count = num_samples - positive_count - negative_count
    for i in range(neutral_count):
        template = random.choice(neutral_templates)
        text = template
        
        # Replace placeholders with random variations
        for placeholder in neutral_words:
            if f'{{{placeholder}}}' in text:
                replacement = random.choice(neutral_words[placeholder])
                text = text.replace(f'{{{placeholder}}}', replacement, 1)
        
        # Add some random variations
        if random.random() < 0.3:
            text += f" {fake.sentence()[:50]}"
        
        texts.append(text)
        labels.append(2)
    
    # Shuffle the dataset
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    return list(texts), list(labels)

def load_training_data():
    """Load training data with enhanced preprocessing."""
    # Generate large dataset using templates and Faker
    texts, labels = generate_sentiment_dataset(num_samples=1000)
    
    logger.info(f"Generated dataset with {len(texts)} samples")
    logger.info(f"Positive samples: {sum(1 for label in labels if label == 1)}")
    logger.info(f"Negative samples: {sum(1 for label in labels if label == 0)}")
    logger.info(f"Neutral samples: {sum(1 for label in labels if label == 2)}")
    
    return texts, labels

def preprocess_texts(texts):
    """Basic text preprocessing."""
    processed = []
    for text in texts:
        # Basic cleaning
        text = text.lower().strip()
        processed.append(text)
    return processed

def train_model():
    """Train the model with hyperparameter tuning and cross-validation."""
    logger.info("Starting model training...")
    
    # Load and preprocess data
    texts, labels = load_training_data()
    texts = preprocess_texts(texts)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Create a pipeline for better model management
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            lowercase=True
        )),
        ('classifier', MultinomialNB())
    ])
    
    # Hyperparameter tuning
    parameters = {
        'vectorizer__max_features': [500, 1000, 1500],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__alpha': [0.1, 0.5, 1.0]
    }
    
    grid_search = GridSearchCV(
        pipeline, parameters, cv=3, scoring='accuracy', n_jobs=-1
    )
    
    # Train with grid search
    logger.info("Performing hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    logger.info(f"Best parameters: {best_params}")
    
    # Cross-validation on the best model
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Print evaluation metrics
    logger.info("\n" + "="*50)
    logger.info("MODEL TRAINING RESULTS")
    logger.info("="*50)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Detailed classification report
    class_names = ['Negative', 'Positive', 'Neutral']
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # Extract vectorizer and model from pipeline
    vectorizer = best_model.named_steps['vectorizer']
    model = best_model.named_steps['classifier']
    
    # Save model components separately
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save metadata
    metadata = {
        'version': '2.0.0',
        'model_type': 'MultinomialNB',
        'feature_extraction': 'TF-IDF',
        'trained_at': datetime.utcnow().isoformat(),
        'test_accuracy': float(test_accuracy),
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'best_params': best_params,
        'classes': class_names,
        'feature_count': len(vectorizer.get_feature_names_out()),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'confusion_matrix': cm.tolist()
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n" + "="*50)
    logger.info("MODEL SAVED SUCCESSFULLY!")
    logger.info(f"Model saved to: {MODEL_PATH}")
    logger.info(f"Vectorizer saved to: {VECTORIZER_PATH}")
    logger.info(f"Metadata saved to: {METADATA_PATH}")
    logger.info("="*50)
    
    return model, vectorizer, metadata

def evaluate_model_quality(model, vectorizer, test_texts, test_labels):
    """Additional model quality evaluation."""
    logger.info("\nPerforming additional quality evaluation...")
    
    # Test on some edge cases
    edge_cases = [
        "",
        "a",
        "This product is... interesting",
        "I have mixed feelings about this",
        "It's not bad but not great either"
    ]
    
    for case in edge_cases:
        if case.strip():  # Skip empty case for prediction
            try:
                vectorized = vectorizer.transform([case])
                pred = model.predict(vectorized)[0]
                prob = model.predict_proba(vectorized)[0].max()
                logger.info(f"Edge case '{case[:30]}...' -> Class {pred}, Confidence: {prob:.4f}")
            except Exception as e:
                logger.warning(f"Could not process edge case: {case[:30]}... Error: {e}")

if __name__ == '__main__':
    try:
        model, vectorizer, metadata = train_model()
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        logger.info("You can now run the API with:")
        logger.info("python app.py")
        logger.info("\nOr for production:")
        logger.info("python app.py --workers 4")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
