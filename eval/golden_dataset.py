golden_dataset = [
    {
        "input": "What active ingredient is in Sivanto Prime and what concentration?",
        "expected_output": "Sivanto Prime contains the active ingredient flupyradifurone at a concentration of 200 g/L.",
    },
    {
        "input": "What's the pre-harvest interval for Sivanto Prime on cotton?",
        "expected_output": "The pre-harvest interval for Sivanto Prime on cotton is 14 days.",
    },
    {
        "input": "How many CFU per gram does Serenade Optimum contain?",
        "expected_output": "Serenade Optimum contains Bacillus subtilis strain QST 713 at a minimum concentration of 1 x 10^9 CFU/g.",
    },
    {
        "input": "What's the glyphosate concentration in Roundup PowerMAX 3?",
        "expected_output": "Roundup PowerMAX 3 contains potassium salt of glyphosate equivalent to 4.5 lb acid equivalent per gallon.",
    },
    {
        "input": "What IRAC group is Sivanto Prime classified under?",
        "expected_output": "Sivanto Prime is classified as IRAC Group 4D.",
    },
    {
        "input": "How does Sivanto Prime compare to neonicotinoids in terms of mode of action and resistance?",
        "expected_output": "Sivanto Prime acts as an agonist at the nicotinic acetylcholine receptor (nAChR) like neonicotinoids, but it has a unique binding profile that provides efficacy against neonicotinoid-resistant pest populations. It belongs to the butenolide chemical class rather than the neonicotinoid class.",
    },
    {
        "input": "How do Sivanto Prime and Roundup PowerMAX 3 compare in terms of soil degradation speed?",
        "expected_output": "Sivanto Prime degrades rapidly in soil with a DT50 of 0.7-2.3 days in aerobic soil, while glyphosate in Roundup PowerMAX 3 has a much longer half-life ranging from 2-197 days with a median DT50 of 47 days.",
    },
    {
        "input": "What's the difference between Serenade Optimum's efficacy on powdery mildew versus early blight?",
        "expected_output": "Serenade Optimum has a Good to Excellent efficacy rating against powdery mildew when applied preventively at 7-14 day intervals, while it only has a Moderate efficacy rating against early blight and should be integrated with other fungicides.",
    },
    {
        "input": "What are the two active ingredients in Emesto Silver and their concentrations?",
        "expected_output": "Emesto Silver combines penflufen at 75 g/L and prothioconazole at 37.5 g/L.",
    },
    {
        "input": "How effective was Emesto Silver against Rhizoctonia in field trials?",
        "expected_output": "In field trials across 15 locations over 3 years, Emesto Silver demonstrated 85-95% reduction in Rhizoctonia stem canker severity compared to untreated controls.",
    },
    {
        "input": "Which products in the guide are approved for use on grapes, and what do they target?",
        "expected_output": "Sivanto Prime is approved for use on grapes targeting leafhoppers at 7.0-10.5 fl oz/acre, and Serenade Optimum is approved for grapes targeting powdery mildew (Erysiphe spp., Podosphaera spp.) and Botrytis gray mold (Botrytis cinerea).",
    },
    {
        "input": "What three mechanisms does Serenade Optimum use to suppress plant diseases?",
        "expected_output": "Serenade Optimum provides disease suppression through direct antagonism, induced systemic resistance (ISR), and competitive exclusion of plant pathogens.",
    },
    {
        "input": "Why can't you use Roundup PowerMAX 3 to control glyphosate-resistant Palmer amaranth?",
        "expected_output": "Glyphosate-resistant Palmer amaranth has a 0% control rating with Roundup PowerMAX 3 at any growth stage, meaning the product provides no control whatsoever against resistant populations.",
    },
    {
        "input": "What tank mix restrictions apply to Serenade Optimum?",
        "expected_output": "Serenade Optimum should not be mixed with strongly alkaline products (pH > 9.0) or products containing chlorine-based sanitizers, as these will reduce the viability of Bacillus subtilis. When tank mixing, Serenade Optimum should be added last and the mixture used within 24 hours.",
    },
    {
        "input": "What weather conditions does Bayer recommend for applying Roundup PowerMAX 3 to minimize environmental impact?",
        "expected_output": "Bayer recommends applying Roundup PowerMAX 3 when wind speed is below 10 mph, temperature is below 85 degrees F, and humidity is above 60% when practical, while maintaining buffer zones of minimum 25 feet from aquatic habitats.",
    },
]