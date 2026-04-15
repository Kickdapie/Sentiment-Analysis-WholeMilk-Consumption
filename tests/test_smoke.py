import unittest

from compare_sentiment_models import vader_label_from_compound
from scraper import clean_text, reddit_post_policy_relevant


class SmokeTests(unittest.TestCase):
    def test_clean_text_removes_urls_and_extra_space(self):
        raw = "Check this out   https://example.com/test   now"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Check this out now")

    def test_reddit_policy_relevance_true_for_milk_policy_text(self):
        text = (
            "USDA school meal policy update allows whole milk and 2% milk "
            "in school cafeterias under the federal lunch program."
        )
        self.assertTrue(reddit_post_policy_relevant(text=text, title="School milk policy update", subreddit="news"))

    def test_reddit_policy_relevance_false_for_generic_school_story(self):
        text = "My kid forgot lunch at school and drank some milk at recess."
        self.assertFalse(reddit_post_policy_relevant(text=text, title="School day story", subreddit="parenting"))

    def test_vader_label_mapping_thresholds(self):
        self.assertEqual(vader_label_from_compound(0.25), "positive")
        self.assertEqual(vader_label_from_compound(-0.25), "negative")
        self.assertEqual(vader_label_from_compound(0.00), "neutral")


if __name__ == "__main__":
    unittest.main()
