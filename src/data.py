"""
Sample Data for Semantic Distillation

Contains sample datasets for testing and demonstration.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# AITA Sample Posts
# ---------------------------------------------------------------------------

SAMPLE_AITA_POSTS = [
    {
        "id": "aita_001",
        "body": "My sister asked me to babysit her kids (5 and 7) for the weekend so she could go on a trip with her boyfriend. I said no because I had plans with my friends that I'd made weeks ago. She called me selfish and said family should come first. My parents are siding with her and saying I should have cancelled my plans. I feel bad but I also don't think I should have to drop everything because she didn't plan ahead. She only asked me two days before.",
        "verdict": "NTA",
    },
    {
        "id": "aita_002",
        "body": "I told my wife's best friend that her new restaurant isn't very good. My wife had been raving about it to be supportive, but when the friend directly asked me what I thought, I said the food was mediocre and overpriced. Now my wife is furious with me for being 'brutally honest' instead of just saying something nice. The friend seemed hurt but thanked me for being honest. My wife says I humiliated her friend and made her look bad for recommending it. I think if someone asks for my opinion they should be prepared for honesty.",
        "verdict": "YTA",
    },
    {
        "id": "aita_003",
        "body": "I (28F) refused to let my mother-in-law redecorate our nursery. She showed up with paint swatches and furniture catalogs without being asked. When I said no thank you, we already have a plan, she started crying and told my husband I was excluding her from her grandchild's life. My husband thinks I should have let her 'help' to keep the peace. I've been dealing with her boundary-crossing for our entire marriage and this was the last straw. I told her firmly that this is our baby and our home.",
        "verdict": "NTA",
    },
    {
        "id": "aita_004",
        "body": "I took my name off the lease and moved out while my roommate was at work. We'd been arguing for months about cleaning, noise, and having guests over. I found a new place and just left a note saying I was done. I know this screws him over financially since he can't afford the apartment alone, but I was miserable and couldn't take it anymore. I didn't give 30 days notice. He's been blowing up my phone calling me a coward. Our mutual friends are split on this.",
        "verdict": "YTA",
    },
    {
        "id": "aita_005",
        "body": "My coworker keeps taking credit for my ideas in meetings. Last week I presented a project plan to our manager and she jumped in saying 'oh yes, we discussed this together and I suggested the timeline.' We had never discussed it. I pulled up my Slack messages showing the original idea was entirely mine and sent them to our manager. Now everyone thinks I'm petty for 'receipting' her. But she's been doing this for six months and I've asked her privately to stop three times.",
        "verdict": "NTA",
    },
    {
        "id": "aita_006",
        "body": "I uninvited my brother from my wedding because he told me he was going to propose to his girlfriend during my reception. I told him that was inappropriate and if he couldn't agree to not do it, he shouldn't come. He says I'm being a bridezilla and that a wedding is about family celebrating love. My parents think I overreacted and should just let him have his moment. My fiancé agrees with me. I feel like I'm going crazy for thinking my own wedding should be about us.",
        "verdict": "NTA",
    },
    {
        "id": "aita_007",
        "body": "I ate my roommate's leftover birthday cake. She had a party last weekend and there was about a third of the cake left in the fridge. After four days I figured it was fair game and ate some. Turns out she was saving it for something. She's really upset and says I should have asked. I think four days in a shared fridge is long enough. It didn't have her name on it or anything. She's been giving me the silent treatment for two days now.",
        "verdict": "YTA",
    },
    {
        "id": "aita_008",
        "body": "I reported my neighbor to the HOA for their barking dog. They have a german shepherd that barks for hours when they're at work. I've talked to them three times, left a polite note, and even offered to help pay for training. Nothing changed. The HOA fined them $200 and now they're telling everyone in the neighborhood I'm a snitch. I work from home and literally cannot do my job with the constant barking. I have recordings showing 4+ hours of continuous barking on multiple days.",
        "verdict": "NTA",
    },
    {
        "id": "aita_009",
        "body": "I refused to give my seat to an elderly woman on the bus. Before you judge - I have an invisible disability (chronic pain condition) and standing for the 45 minute ride would have left me in agony. When she asked and I said I couldn't, another passenger started berating me loudly. I didn't want to explain my medical condition to a bus full of strangers. The woman ended up standing and gave me dirty looks the whole ride. I feel terrible but I also shouldn't have to disclose my health issues.",
        "verdict": "NTA",
    },
    {
        "id": "aita_010",
        "body": "I went through my teenage daughter's phone after she started acting secretive and found out she's been dating a 22 year old. She's 16. I immediately told him to stay away from her and grounded her. She says I violated her privacy and she hates me. My ex-wife thinks I should have talked to our daughter first before contacting the guy. I don't care if she hates me right now - a 22 year old has no business dating a 16 year old. I'm considering involving the police.",
        "verdict": "NTA",
    },
    {
        "id": "aita_011",
        "body": "I changed the Netflix password and kicked my ex off the account. We broke up three months ago and she's been using my Netflix, Hulu, and Spotify this whole time. When I finally changed the passwords she texted me saying it was a 'petty' thing to do and that I promised she could keep using them. I never made that promise - I just hadn't gotten around to changing them. She told our friend group I'm being vindictive.",
        "verdict": "NTA",
    },
    {
        "id": "aita_012",
        "body": "I told my friend I don't want to hear about her MLM business anymore. Every time we hang out she tries to recruit me or get me to buy her essential oils. I've bought stuff three times to be supportive but I can't keep spending money on things I don't need. When I finally said 'I value our friendship but I'm not interested in the business side,' she said I wasn't being a real friend and that real friends support each other's dreams. We've been friends for 15 years.",
        "verdict": "NTA",
    },
]


def get_sample_aita_data() -> pd.DataFrame:
    """
    Get the sample AITA posts as a DataFrame.
    
    Returns:
        DataFrame with columns: id, body, verdict
    """
    return pd.DataFrame(SAMPLE_AITA_POSTS)


def load_data(
    path: str | None = None,
    text_col: str = "body",
    label_col: str = "verdict",
    binary_labels: bool = True,
) -> pd.DataFrame:
    """
    Load data from CSV or return sample data if no path provided.
    
    Args:
        path: Path to CSV file, or None to use sample data
        text_col: Name of the text column (for validation)
        label_col: Name of the label column (for validation)
        binary_labels: If True, convert to binary NTA/YTA classification:
                       - NAH (No Assholes Here) → NTA (poster is not at fault)
                       - ESH (Everyone Sucks Here) → dropped (ambiguous)
    
    Returns:
        DataFrame with the loaded data
    """
    if path is None:
        return get_sample_aita_data()
    
    df = pd.read_csv(path)
    
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in data")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in data")
    
    if binary_labels and label_col in df.columns:
        original_len = len(df)
        
        # Map NAH → NTA (No Assholes Here means poster is not at fault)
        df[label_col] = df[label_col].replace({"NAH": "NTA"})
        
        # Drop ESH (Everyone Sucks Here is ambiguous for binary classification)
        df = df[df[label_col] != "ESH"].reset_index(drop=True)
        
        dropped = original_len - len(df)
        if dropped > 0:
            print(f"Binary labels: mapped NAH→NTA, dropped {dropped} ESH samples ({dropped/original_len*100:.1f}%)")
    
    return df
