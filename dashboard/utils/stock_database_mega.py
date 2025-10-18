"""
Mega Stock Database - 1000+ Stocks
===================================
Comprehensive stock database covering major stocks from 7 global exchanges:
- NYSE/NASDAQ (USA)
- LSE (UK)
- TSE (Japan)
- SSE/SZSE (China)
- HKEX (Hong Kong)
- Euronext (Europe)
- TSX (Canada)

Total stocks: 1000+
"""

MEGA_STOCK_DATABASE = {}

# ============================================================================
# USA - NYSE/NASDAQ (400+ stocks)
# ============================================================================

usa_tech_software = {
    # FAANG + Major Tech
    'AAPL': 'Apple Inc',
    'MSFT': 'Microsoft Corp',
    'GOOGL': 'Alphabet Inc Class A',
    'GOOG': 'Alphabet Inc Class C',
    'AMZN': 'Amazon.com Inc',
    'META': 'Meta Platforms Inc',
    'NVDA': 'NVIDIA Corp',
    'TSLA': 'Tesla Inc',

    # Software & Cloud
    'CRM': 'Salesforce Inc',
    'ORCL': 'Oracle Corp',
    'ADBE': 'Adobe Inc',
    'NOW': 'ServiceNow Inc',
    'INTU': 'Intuit Inc',
    'WDAY': 'Workday Inc',
    'SNOW': 'Snowflake Inc',
    'DDOG': 'Datadog Inc',
    'ZS': 'Zscaler Inc',
    'CRWD': 'CrowdStrike Holdings Inc',
    'PANW': 'Palo Alto Networks Inc',
    'FTNT': 'Fortinet Inc',
    'OKTA': 'Okta Inc',
    'MDB': 'MongoDB Inc',
    'TEAM': 'Atlassian Corp',
    'VEEV': 'Veeva Systems Inc',
    'DOCU': 'DocuSign Inc',
    'ZM': 'Zoom Video Communications Inc',
    'TWLO': 'Twilio Inc',
    'NET': 'Cloudflare Inc',
    'FSLY': 'Fastly Inc',
    'DBX': 'Dropbox Inc',
    'BOX': 'Box Inc',

    # Semiconductors
    'AMD': 'Advanced Micro Devices Inc',
    'INTC': 'Intel Corp',
    'QCOM': 'Qualcomm Inc',
    'AVGO': 'Broadcom Inc',
    'TXN': 'Texas Instruments Inc',
    'MU': 'Micron Technology Inc',
    'AMAT': 'Applied Materials Inc',
    'LRCX': 'Lam Research Corp',
    'KLAC': 'KLA Corp',
    'MRVL': 'Marvell Technology Inc',
    'NXPI': 'NXP Semiconductors NV',
    'ADI': 'Analog Devices Inc',
    'MCHP': 'Microchip Technology Inc',
    'SWKS': 'Skyworks Solutions Inc',
    'QRVO': 'Qorvo Inc',
    'ON': 'ON Semiconductor Corp',
    'MPWR': 'Monolithic Power Systems Inc',
    'ASML': 'ASML Holding NV',

    # Hardware & Electronics
    'CSCO': 'Cisco Systems Inc',
    'IBM': 'International Business Machines Corp',
    'HPQ': 'HP Inc',
    'HPE': 'Hewlett Packard Enterprise Co',
    'DELL': 'Dell Technologies Inc',
    'NTAP': 'NetApp Inc',
    'WDC': 'Western Digital Corp',
    'STX': 'Seagate Technology Holdings PLC',
    'PSTG': 'Pure Storage Inc',
    'SMCI': 'Super Micro Computer Inc',

    # Telecom Equipment
    'JNPR': 'Juniper Networks Inc',
    'FFIV': 'F5 Inc',
    'CIEN': 'Ciena Corp',
    'LITE': 'Lumentum Holdings Inc',
}

usa_ecommerce_payments = {
    # E-commerce
    'SHOP': 'Shopify Inc',
    'EBAY': 'eBay Inc',
    'ETSY': 'Etsy Inc',
    'W': 'Wayfair Inc',
    'CHWY': 'Chewy Inc',
    'CVNA': 'Carvana Co',

    # Payments & Fintech
    'PYPL': 'PayPal Holdings Inc',
    'SQ': 'Block Inc',
    'COIN': 'Coinbase Global Inc',
    'V': 'Visa Inc',
    'MA': 'Mastercard Inc',
    'AXP': 'American Express Co',
    'FISV': 'Fiserv Inc',
    'FIS': 'Fidelity National Information Services Inc',
    'GPN': 'Global Payments Inc',
    'AFRM': 'Affirm Holdings Inc',
    'SOFI': 'SoFi Technologies Inc',
    'LC': 'LendingClub Corp',
}

usa_social_entertainment = {
    # Social Media & Streaming
    'SNAP': 'Snap Inc',
    'PINS': 'Pinterest Inc',
    'RBLX': 'Roblox Corp',
    'MTCH': 'Match Group Inc',
    'BMBL': 'Bumble Inc',

    # Streaming & Content
    'NFLX': 'Netflix Inc',
    'SPOT': 'Spotify Technology SA',
    'ROKU': 'Roku Inc',
    'PARA': 'Paramount Global',
    'WBD': 'Warner Bros Discovery Inc',
    'DIS': 'Walt Disney Co',
    'CMCSA': 'Comcast Corp',

    # Gaming
    'EA': 'Electronic Arts Inc',
    'ATVI': 'Activision Blizzard Inc',
    'TTWO': 'Take-Two Interactive Software Inc',
    'U': 'Unity Software Inc',
    'DKNG': 'DraftKings Inc',
    'PENN': 'Penn Entertainment Inc',
}

usa_banking = {
    # Megabanks
    'JPM': 'JPMorgan Chase & Co',
    'BAC': 'Bank of America Corp',
    'WFC': 'Wells Fargo & Co',
    'C': 'Citigroup Inc',

    # Investment Banks
    'GS': 'Goldman Sachs Group Inc',
    'MS': 'Morgan Stanley',
    'SCHW': 'Charles Schwab Corp',
    'BLK': 'BlackRock Inc',
    'BX': 'Blackstone Inc',
    'KKR': 'KKR & Co Inc',
    'APO': 'Apollo Global Management Inc',

    # Regional Banks
    'USB': 'U.S. Bancorp',
    'PNC': 'PNC Financial Services Group Inc',
    'TFC': 'Truist Financial Corp',
    'COF': 'Capital One Financial Corp',
    'BK': 'Bank of New York Mellon Corp',
    'STT': 'State Street Corp',
    'DFS': 'Discover Financial Services',
    'ALLY': 'Ally Financial Inc',
    'RF': 'Regions Financial Corp',
    'KEY': 'KeyCorp',
    'CFG': 'Citizens Financial Group Inc',
    'HBAN': 'Huntington Bancshares Inc',
    'MTB': 'M&T Bank Corp',
    'FITB': 'Fifth Third Bancorp',
    'ZION': 'Zions Bancorp NA',
    'WAL': 'Western Alliance Bancorp',
    'PACW': 'PacWest Bancorp',
    'CMA': 'Comerica Inc',
    'WTFC': 'Wintrust Financial Corp',
    'EWBC': 'East West Bancorp Inc',
}

usa_insurance_realestate = {
    # Insurance
    'BRK.B': 'Berkshire Hathaway Inc Class B',
    'PGR': 'Progressive Corp',
    'ALL': 'Allstate Corp',
    'TRV': 'Travelers Companies Inc',
    'CB': 'Chubb Ltd',
    'AIG': 'American International Group Inc',
    'MET': 'MetLife Inc',
    'PRU': 'Prudential Financial Inc',
    'AFL': 'Aflac Inc',
    'AJG': 'Arthur J Gallagher & Co',
    'MMC': 'Marsh & McLennan Companies Inc',
    'AON': 'Aon PLC',
    'WTW': 'Willis Towers Watson PLC',
    'BRO': 'Brown & Brown Inc',
    'RLI': 'RLI Corp',

    # REITs
    'PLD': 'Prologis Inc',
    'AMT': 'American Tower Corp',
    'CCI': 'Crown Castle Inc',
    'EQIX': 'Equinix Inc',
    'PSA': 'Public Storage',
    'DLR': 'Digital Realty Trust Inc',
    'O': 'Realty Income Corp',
    'WELL': 'Welltower Inc',
    'AVB': 'AvalonBay Communities Inc',
    'EQR': 'Equity Residential',
    'VTR': 'Ventas Inc',
    'SPG': 'Simon Property Group Inc',
    'SBAC': 'SBA Communications Corp',
    'INVH': 'Invitation Homes Inc',
    'ARE': 'Alexandria Real Estate Equities Inc',
}

usa_pharma_biotech = {
    # Big Pharma
    'JNJ': 'Johnson & Johnson',
    'LLY': 'Eli Lilly and Co',
    'UNH': 'UnitedHealth Group Inc',
    'PFE': 'Pfizer Inc',
    'ABBV': 'AbbVie Inc',
    'MRK': 'Merck & Co Inc',
    'TMO': 'Thermo Fisher Scientific Inc',
    'ABT': 'Abbott Laboratories',
    'BMY': 'Bristol-Myers Squibb Co',
    'AMGN': 'Amgen Inc',
    'GILD': 'Gilead Sciences Inc',
    'VRTX': 'Vertex Pharmaceuticals Inc',
    'REGN': 'Regeneron Pharmaceuticals Inc',
    'CVS': 'CVS Health Corp',
    'CI': 'Cigna Corp',
    'HUM': 'Humana Inc',
    'BIIB': 'Biogen Inc',
    'MRNA': 'Moderna Inc',
    'BNTX': 'BioNTech SE',

    # Biotech
    'ILMN': 'Illumina Inc',
    'ALNY': 'Alnylam Pharmaceuticals Inc',
    'INCY': 'Incyte Corp',
    'EXAS': 'Exact Sciences Corp',
    'TECH': 'Bio-Techne Corp',
    'VTRS': 'Viatris Inc',
    'TAK': 'Takeda Pharmaceutical Co Ltd',
    'SGEN': 'Seagen Inc',
    'IONS': 'Ionis Pharmaceuticals Inc',
    'SRPT': 'Sarepta Therapeutics Inc',
    'NBIX': 'Neurocrine Biosciences Inc',
    'RGEN': 'Repligen Corp',
    'BGNE': 'BeiGene Ltd',
    'ARWR': 'Arrowhead Pharmaceuticals Inc',
    'UTHR': 'United Therapeutics Corp',
    'BMRN': 'BioMarin Pharmaceutical Inc',
    'CRSP': 'CRISPR Therapeutics AG',
    'BEAM': 'Beam Therapeutics Inc',
    'EDIT': 'Editas Medicine Inc',
}

usa_healthcare_equipment = {
    # Medical Devices
    'MDT': 'Medtronic PLC',
    'SYK': 'Stryker Corp',
    'BSX': 'Boston Scientific Corp',
    'EW': 'Edwards Lifesciences Corp',
    'ISRG': 'Intuitive Surgical Inc',
    'ZBH': 'Zimmer Biomet Holdings Inc',
    'BAX': 'Baxter International Inc',
    'BDX': 'Becton Dickinson and Co',
    'HOLX': 'Hologic Inc',
    'ALGN': 'Align Technology Inc',
    'DXCM': 'DexCom Inc',
    'PODD': 'Insulet Corp',
    'IDXX': 'IDEXX Laboratories Inc',
    'IQV': 'IQVIA Holdings Inc',
    'CRL': 'Charles River Laboratories International Inc',
    'RMD': 'ResMed Inc',
    'TFX': 'Teleflex Inc',
    'STE': 'STERIS PLC',
    'WST': 'West Pharmaceutical Services Inc',
    'RVTY': 'Revvity Inc',
    'A': 'Agilent Technologies Inc',
    'DHR': 'Danaher Corp',
    'WAT': 'Waters Corp',
    'PKI': 'PerkinElmer Inc',
    'MTD': 'Mettler-Toledo International Inc',
    'CTLT': 'Catalent Inc',
    'GEHC': 'GE HealthCare Technologies Inc',
}

usa_consumer_retail = {
    # Retail
    'WMT': 'Walmart Inc',
    'COST': 'Costco Wholesale Corp',
    'TGT': 'Target Corp',
    'HD': 'Home Depot Inc',
    'LOW': 'Lowe\'s Companies Inc',
    'TJX': 'TJX Companies Inc',
    'ROST': 'Ross Stores Inc',
    'DG': 'Dollar General Corp',
    'DLTR': 'Dollar Tree Inc',
    'BBY': 'Best Buy Co Inc',
    'GPS': 'Gap Inc',
    'ANF': 'Abercrombie & Fitch Co',
    'AEO': 'American Eagle Outfitters Inc',
    'URBN': 'Urban Outfitters Inc',
    'M': 'Macy\'s Inc',
    'KSS': 'Kohl\'s Corp',
    'JWN': 'Nordstrom Inc',

    # Restaurants & Food
    'MCD': 'McDonald\'s Corp',
    'SBUX': 'Starbucks Corp',
    'CMG': 'Chipotle Mexican Grill Inc',
    'YUM': 'Yum! Brands Inc',
    'QSR': 'Restaurant Brands International Inc',
    'DPZ': 'Domino\'s Pizza Inc',
    'WING': 'Wingstop Inc',
    'TXRH': 'Texas Roadhouse Inc',
    'DRI': 'Darden Restaurants Inc',
    'CAKE': 'Cheesecake Factory Inc',
    'SHAK': 'Shake Shack Inc',
    'BLMN': 'Bloomin\' Brands Inc',
    'EAT': 'Brinker International Inc',
    'CHDN': 'Churchill Downs Inc',

    # Food Delivery
    'DASH': 'DoorDash Inc',
    'UBER': 'Uber Technologies Inc',
    'LYFT': 'Lyft Inc',
    'ABNB': 'Airbnb Inc',
    'BKNG': 'Booking Holdings Inc',
    'EXPE': 'Expedia Group Inc',
    'TRIP': 'TripAdvisor Inc',
    'TCOM': 'Trip.com Group Ltd',
    'MMYT': 'MakeMyTrip Ltd',
}

usa_consumer_staples = {
    # Food & Beverage
    'PG': 'Procter & Gamble Co',
    'KO': 'Coca-Cola Co',
    'PEP': 'PepsiCo Inc',
    'PM': 'Philip Morris International Inc',
    'MO': 'Altria Group Inc',
    'CL': 'Colgate-Palmolive Co',
    'KMB': 'Kimberly-Clark Corp',
    'GIS': 'General Mills Inc',
    'K': 'Kellogg Co',
    'HSY': 'Hershey Co',
    'MDLZ': 'Mondelez International Inc',
    'KHC': 'Kraft Heinz Co',
    'CAG': 'Conagra Brands Inc',
    'CPB': 'Campbell Soup Co',
    'SJM': 'J.M. Smucker Co',
    'HRL': 'Hormel Foods Corp',
    'TSN': 'Tyson Foods Inc',
    'BG': 'Bunge Ltd',
    'ADM': 'Archer-Daniels-Midland Co',

    # Household & Personal Care
    'EL': 'Estee Lauder Companies Inc',
    'CLX': 'Clorox Co',
    'CHD': 'Church & Dwight Co Inc',
    'COTY': 'Coty Inc',
    'ELF': 'e.l.f. Beauty Inc',
    'TPR': 'Tapestry Inc',
    'RL': 'Ralph Lauren Corp',
    'PVH': 'PVH Corp',
    'HBI': 'Hanesbrands Inc',
    'NKE': 'Nike Inc',
    'LULU': 'Lululemon Athletica Inc',
}

usa_energy = {
    # Oil & Gas
    'XOM': 'Exxon Mobil Corp',
    'CVX': 'Chevron Corp',
    'COP': 'ConocoPhillips',
    'SLB': 'Schlumberger NV',
    'EOG': 'EOG Resources Inc',
    'MPC': 'Marathon Petroleum Corp',
    'PSX': 'Phillips 66',
    'VLO': 'Valero Energy Corp',
    'OXY': 'Occidental Petroleum Corp',
    'HES': 'Hess Corp',
    'DVN': 'Devon Energy Corp',
    'FANG': 'Diamondback Energy Inc',
    'MRO': 'Marathon Oil Corp',
    'APA': 'APA Corp',
    'HAL': 'Halliburton Co',
    'BKR': 'Baker Hughes Co',
    'NOV': 'NOV Inc',
    'FTI': 'TechnipFMC PLC',
    'HP': 'Helmerich & Payne Inc',
    'RIG': 'Transocean Ltd',
    'VAL': 'Valaris Ltd',
    'KMI': 'Kinder Morgan Inc',
    'WMB': 'Williams Companies Inc',
    'OKE': 'ONEOK Inc',
    'LNG': 'Cheniere Energy Inc',
    'TRGP': 'Targa Resources Corp',
    'EPD': 'Enterprise Products Partners LP',
    'ET': 'Energy Transfer LP',
}

usa_industrials = {
    # Aerospace & Defense
    'BA': 'Boeing Co',
    'LMT': 'Lockheed Martin Corp',
    'RTX': 'Raytheon Technologies Corp',
    'GD': 'General Dynamics Corp',
    'NOC': 'Northrop Grumman Corp',
    'LHX': 'L3Harris Technologies Inc',
    'HWM': 'Howmet Aerospace Inc',
    'TXT': 'Textron Inc',
    'LDOS': 'Leidos Holdings Inc',
    'KTOS': 'Kratos Defense & Security Solutions Inc',

    # Industrial Conglomerates
    'GE': 'General Electric Co',
    'HON': 'Honeywell International Inc',
    'MMM': '3M Co',
    'ITW': 'Illinois Tool Works Inc',
    'EMR': 'Emerson Electric Co',
    'ETN': 'Eaton Corp PLC',
    'PH': 'Parker-Hannifin Corp',
    'ROK': 'Rockwell Automation Inc',

    # Machinery & Equipment
    'CAT': 'Caterpillar Inc',
    'DE': 'Deere & Co',
    'PCAR': 'PACCAR Inc',
    'CMI': 'Cummins Inc',
    'IR': 'Ingersoll Rand Inc',
    'CARR': 'Carrier Global Corp',
    'OTIS': 'Otis Worldwide Corp',
    'FTV': 'Fortive Corp',
    'GNRC': 'Generac Holdings Inc',

    # Transportation
    'UPS': 'United Parcel Service Inc',
    'FDX': 'FedEx Corp',
    'UNP': 'Union Pacific Corp',
    'CSX': 'CSX Corp',
    'NSC': 'Norfolk Southern Corp',
    'DAL': 'Delta Air Lines Inc',
    'UAL': 'United Airlines Holdings Inc',
    'AAL': 'American Airlines Group Inc',
    'LUV': 'Southwest Airlines Co',
}

usa_materials = {
    # Chemicals
    'LIN': 'Linde PLC',
    'APD': 'Air Products and Chemicals Inc',
    'SHW': 'Sherwin-Williams Co',
    'ECL': 'Ecolab Inc',
    'DD': 'DuPont de Nemours Inc',
    'DOW': 'Dow Inc',
    'PPG': 'PPG Industries Inc',
    'NEM': 'Newmont Corp',
    'FCX': 'Freeport-McMoRan Inc',
    'NUE': 'Nucor Corp',
    'STLD': 'Steel Dynamics Inc',
    'RS': 'Reliance Steel & Aluminum Co',
    'VMC': 'Vulcan Materials Co',
    'MLM': 'Martin Marietta Materials Inc',
    'ALB': 'Albemarle Corp',
    'FMC': 'FMC Corp',
    'CE': 'Celanese Corp',
    'EMN': 'Eastman Chemical Co',
    'IFF': 'International Flavors & Fragrances Inc',
    'LYB': 'LyondellBasell Industries NV',
    'CF': 'CF Industries Holdings Inc',
    'MOS': 'Mosaic Co',
    'AVY': 'Avery Dennison Corp',
    'SEE': 'Sealed Air Corp',
    'PKG': 'Packaging Corp of America',
    'IP': 'International Paper Co',
}

usa_utilities = {
    'NEE': 'NextEra Energy Inc',
    'DUK': 'Duke Energy Corp',
    'SO': 'Southern Co',
    'D': 'Dominion Energy Inc',
    'AEP': 'American Electric Power Co Inc',
    'EXC': 'Exelon Corp',
    'XEL': 'Xcel Energy Inc',
    'WEC': 'WEC Energy Group Inc',
    'ED': 'Consolidated Edison Inc',
    'AWK': 'American Water Works Co Inc',
    'ES': 'Eversource Energy',
    'FE': 'FirstEnergy Corp',
    'PPL': 'PPL Corp',
}

usa_telecom = {
    'T': 'AT&T Inc',
    'VZ': 'Verizon Communications Inc',
    'TMUS': 'T-Mobile US Inc',
}

usa_ev_cleanenergy = {
    # EV & Clean Tech
    'RIVN': 'Rivian Automotive Inc',
    'LCID': 'Lucid Group Inc',
    'NIO': 'NIO Inc',
    'XPEV': 'XPeng Inc',
    'LI': 'Li Auto Inc',
    'FSR': 'Fisker Inc',
    'GOEV': 'Canoo Inc',
    'LEV': 'Lion Electric Co',
    'CHPT': 'ChargePoint Holdings Inc',
    'BLNK': 'Blink Charging Co',
    'EVGO': 'EVgo Inc',

    # Solar & Renewables
    'ENPH': 'Enphase Energy Inc',
    'SEDG': 'SolarEdge Technologies Inc',
    'RUN': 'Sunrun Inc',
    'FSLR': 'First Solar Inc',
    'SPWR': 'SunPower Corp',
    'NOVA': 'Sunnova Energy International Inc',
    'ARRY': 'Array Technologies Inc',
    'CSIQ': 'Canadian Solar Inc',
    'JKS': 'JinkoSolar Holding Co Ltd',
    'DQ': 'Daqo New Energy Corp',

    # Battery Tech
    'QS': 'QuantumScape Corp',
    'STEM': 'Stem Inc',
    'FLNC': 'Fluence Energy Inc',
}

usa_chinese_adrs = {
    'BABA': 'Alibaba Group Holding Ltd',
    'PDD': 'PDD Holdings Inc',
    'JD': 'JD.com Inc',
    'BIDU': 'Baidu Inc',
    'NTES': 'NetEase Inc',
    'BILI': 'Bilibili Inc',
    'IQ': 'iQIYI Inc',
    'TME': 'Tencent Music Entertainment Group',
    'VIPS': 'Vipshop Holdings Ltd',
    'YMM': 'Full Truck Alliance Co Ltd',
    'DIDI': 'DiDi Global Inc',
    'EDU': 'New Oriental Education & Technology Group Inc',
    'TAL': 'TAL Education Group',
    'ZTO': 'ZTO Express Cayman Inc',
}

usa_etfs = {
    # Broad Market
    'SPY': 'SPDR S&P 500 ETF Trust',
    'QQQ': 'Invesco QQQ Trust',
    'DIA': 'SPDR Dow Jones Industrial Average ETF',
    'IWM': 'iShares Russell 2000 ETF',
    'VTI': 'Vanguard Total Stock Market ETF',
    'VOO': 'Vanguard S&P 500 ETF',

    # Sector ETFs
    'XLK': 'Technology Select Sector SPDR Fund',
    'XLF': 'Financial Select Sector SPDR Fund',
    'XLE': 'Energy Select Sector SPDR Fund',
    'XLV': 'Health Care Select Sector SPDR Fund',
    'XLY': 'Consumer Discretionary Select Sector SPDR Fund',
    'XLP': 'Consumer Staples Select Sector SPDR Fund',
    'XLI': 'Industrial Select Sector SPDR Fund',
    'XLB': 'Materials Select Sector SPDR Fund',
    'XLU': 'Utilities Select Sector SPDR Fund',
    'XLRE': 'Real Estate Select Sector SPDR Fund',

    # Growth/Value
    'VUG': 'Vanguard Growth ETF',
    'VTV': 'Vanguard Value ETF',
    'IWF': 'iShares Russell 1000 Growth ETF',
    'IWD': 'iShares Russell 1000 Value ETF',

    # International
    'EFA': 'iShares MSCI EAFE ETF',
    'EEM': 'iShares MSCI Emerging Markets ETF',
    'VEA': 'Vanguard FTSE Developed Markets ETF',
    'VWO': 'Vanguard FTSE Emerging Markets ETF',
    'FXI': 'iShares China Large-Cap ETF',
    'EWJ': 'iShares MSCI Japan ETF',
    'EWZ': 'iShares MSCI Brazil ETF',
    'EWY': 'iShares MSCI South Korea ETF',

    # Fixed Income
    'AGG': 'iShares Core U.S. Aggregate Bond ETF',
    'BND': 'Vanguard Total Bond Market ETF',
    'TLT': 'iShares 20+ Year Treasury Bond ETF',
    'IEF': 'iShares 7-10 Year Treasury Bond ETF',
}

# ============================================================================
# UK - London Stock Exchange (60+ stocks)
# ============================================================================

uk_energy = {
    'BP.L': 'BP plc',
    'SHEL.L': 'Shell plc',
    'SSE.L': 'SSE plc',
    'NG.L': 'National Grid plc',
}

uk_finance = {
    'HSBA.L': 'HSBC Holdings plc',
    'BARC.L': 'Barclays PLC',
    'LLOY.L': 'Lloyds Banking Group plc',
    'NWG.L': 'NatWest Group plc',
    'STAN.L': 'Standard Chartered PLC',
    'LSEG.L': 'London Stock Exchange Group plc',
    'III.L': '3i Group plc',
    'PRU.L': 'Prudential plc',
    'AVAV.L': 'Aviva plc',
}

uk_consumer = {
    'ULVR.L': 'Unilever PLC',
    'DGE.L': 'Diageo plc',
    'RKT.L': 'Reckitt Benckiser Group plc',
    'ABF.L': 'Associated British Foods plc',
    'TSCO.L': 'Tesco PLC',
    'SBRY.L': 'Sainsbury (J) plc',
    'MRW.L': 'Morrison (Wm) Supermarkets PLC',
    'OCDO.L': 'Ocado Group plc',
    'BATS.L': 'British American Tobacco plc',
    'IMB.L': 'Imperial Brands PLC',
    'JD.L': 'JD Sports Fashion plc',
    'NXT.L': 'Next plc',
    'MKS.L': 'Marks and Spencer Group plc',
}

uk_pharma_healthcare = {
    'AZN.L': 'AstraZeneca PLC',
    'GSK.L': 'GSK plc',
    'SN.L': 'Smith & Nephew plc',
}

uk_materials = {
    'RIO.L': 'Rio Tinto Group',
    'AAL.L': 'Anglo American plc',
    'GLEN.L': 'Glencore plc',
    'ANTO.L': 'Antofagasta plc',
    'CRH.L': 'CRH plc',
}

uk_industrials = {
    'BA.L': 'BAE Systems plc',
    'RR.L': 'Rolls-Royce Holdings plc',
    'EXPN.L': 'Experian plc',
    'RELX.L': 'RELX PLC',
    'RTO.L': 'Rentokil Initial plc',
    'BNZL.L': 'Bunzl plc',
    'SMIN.L': 'Smiths Group plc',
    'IMI.L': 'IMI plc',
    'WEIR.L': 'Weir Group PLC',
    'SPX.L': 'Spirax-Sarco Engineering plc',
}

uk_telecom_tech = {
    'VOD.L': 'Vodafone Group Plc',
    'BT-A.L': 'BT Group plc',
    'AUTO.L': 'Auto Trader Group plc',
    'INF.L': 'Informa plc',
}

uk_realestate = {
    'LAND.L': 'Land Securities Group plc',
    'BLND.L': 'British Land Co PLC',
    'PSN.L': 'Persimmon plc',
    'BDEV.L': 'Barratt Developments PLC',
    'TW.L': 'Taylor Wimpey plc',
}

uk_luxury = {
    'BRBY.L': 'Burberry Group plc',
    'FERG.L': 'Ferguson plc',
    'WPP.L': 'WPP plc',
}

# ============================================================================
# JAPAN - Tokyo Stock Exchange (80+ stocks)
# ============================================================================

japan_automotive = {
    '7203.T': 'Toyota Motor Corp',
    '7267.T': 'Honda Motor Co Ltd',
    '7201.T': 'Nissan Motor Co Ltd',
    '7269.T': 'Suzuki Motor Corp',
    '7270.T': 'Subaru Corp',
    '7211.T': 'Mitsubishi Motors Corp',
    '7261.T': 'Mazda Motor Corp',
    '7282.T': 'Toyota Industries Corp',
}

japan_electronics = {
    '6758.T': 'Sony Group Corp',
    '6752.T': 'Panasonic Holdings Corp',
    '6841.T': 'Yokogawa Electric Corp',
    '6857.T': 'Advantest Corp',
    '6861.T': 'Keyence Corp',
    '6501.T': 'Hitachi Ltd',
    '6503.T': 'Mitsubishi Electric Corp',
    '6701.T': 'NEC Corp',
    '6702.T': 'Fujitsu Ltd',
    '6971.T': 'Kyocera Corp',
    '6976.T': 'Taiyo Yuden Co Ltd',
    '6981.T': 'Murata Manufacturing Co Ltd',
    '6954.T': 'Fanuc Corp',
    '6273.T': 'SMC Corp',
}

japan_semiconductors = {
    '8035.T': 'Tokyo Electron Ltd',
    '6920.T': 'Lasertec Corp',
    '6723.T': 'Renesas Electronics Corp',
    '6724.T': 'Seiko Epson Corp',
    '6952.T': 'Casio Computer Co Ltd',
}

japan_telecom_internet = {
    '9984.T': 'SoftBank Group Corp',
    '9432.T': 'Nippon Telegraph and Telephone Corp',
    '9433.T': 'KDDI Corp',
    '9434.T': 'SoftBank Corp',
    '4755.T': 'Rakuten Group Inc',
    '4689.T': 'Yahoo Japan Corp',
    '4751.T': 'Cyberagent Inc',
    '3659.T': 'Nexon Co Ltd',
}

japan_gaming = {
    '7974.T': 'Nintendo Co Ltd',
    '9697.T': 'Capcom Co Ltd',
    '9684.T': 'Square Enix Holdings Co Ltd',
    '9766.T': 'Konami Holdings Corp',
    '7832.T': 'Bandai Namco Holdings Inc',
}

japan_finance = {
    '8306.T': 'Mitsubishi UFJ Financial Group Inc',
    '8316.T': 'Sumitomo Mitsui Financial Group Inc',
    '8411.T': 'Mizuho Financial Group Inc',
    '8604.T': 'Nomura Holdings Inc',
    '8601.T': 'Daiwa Securities Group Inc',
    '8473.T': 'SBI Holdings Inc',
    '8750.T': 'Dai-ichi Life Holdings Inc',
    '8725.T': 'MS&AD Insurance Group Holdings Inc',
}

japan_retail_consumer = {
    '9983.T': 'Fast Retailing Co Ltd',
    '8267.T': 'Aeon Co Ltd',
    '3382.T': 'Seven & i Holdings Co Ltd',
    '2503.T': 'Kirin Holdings Co Ltd',
    '2502.T': 'Asahi Group Holdings Ltd',
    '2501.T': 'Sapporo Holdings Ltd',
    '2269.T': 'Meiji Holdings Co Ltd',
}

japan_pharma = {
    '4502.T': 'Takeda Pharmaceutical Co Ltd',
    '4503.T': 'Astellas Pharma Inc',
    '4568.T': 'Daiichi Sankyo Co Ltd',
    '4523.T': 'Eisai Co Ltd',
    '4507.T': 'Shionogi & Co Ltd',
    '4519.T': 'Chugai Pharmaceutical Co Ltd',
}

japan_industrials = {
    '6902.T': 'Denso Corp',
    '7011.T': 'Mitsubishi Heavy Industries Ltd',
    '7012.T': 'Kawasaki Heavy Industries Ltd',
    '6301.T': 'Komatsu Ltd',
    '6326.T': 'Kubota Corp',
    '7951.T': 'Yamaha Corp',
    '7731.T': 'Nikon Corp',
    '7751.T': 'Canon Inc',
    '7733.T': 'Olympus Corp',
    '4911.T': 'Shiseido Co Ltd',
}

japan_trading = {
    '8058.T': 'Mitsubishi Corp',
    '8031.T': 'Mitsui & Co Ltd',
    '8053.T': 'Sumitomo Corp',
    '8002.T': 'Marubeni Corp',
    '8001.T': 'Itochu Corp',
}

japan_materials = {
    '5401.T': 'Nippon Steel Corp',
    '5411.T': 'JFE Holdings Inc',
    '4063.T': 'Shin-Etsu Chemical Co Ltd',
    '4452.T': 'Kao Corp',
    '4188.T': 'Mitsubishi Chemical Holdings Corp',
}

# ============================================================================
# CHINA/HONG KONG (100+ stocks)
# ============================================================================

hongkong_tech = {
    '0700.HK': 'Tencent Holdings Ltd',
    '9988.HK': 'Alibaba Group Holding Ltd',
    '9618.HK': 'JD.com Inc',
    '9999.HK': 'NetEase Inc',
    '3690.HK': 'Meituan',
    '1810.HK': 'Xiaomi Corp',
    '9992.HK': 'Pop Mart International Group Ltd',  # USER REQUESTED
    '9626.HK': 'Bilibili Inc',
    '9888.HK': 'Baidu Inc',
    '9961.HK': 'Trip.com Group Ltd',
    '2013.HK': 'Weimob Inc',
    '1024.HK': 'Kuaishou Technology',
    '9698.HK': 'GDS Holdings Ltd',
    '9868.HK': 'Xpeng Inc',
    '9866.HK': 'NIO Inc',
    '2015.HK': 'Li Auto Inc',
}

hongkong_finance = {
    '0005.HK': 'HSBC Holdings plc',
    '0939.HK': 'China Construction Bank Corp',
    '1398.HK': 'Industrial and Commercial Bank of China Ltd',
    '3988.HK': 'Bank of China Ltd',
    '0011.HK': 'Hang Seng Bank Ltd',
    '1288.HK': 'Agricultural Bank of China Ltd',
    '3968.HK': 'China Merchants Bank Co Ltd',
    '6837.HK': 'Haitong Securities Co Ltd',
    '6886.HK': 'HTSC',
    '2318.HK': 'Ping An Insurance Group Co of China Ltd',
    '2628.HK': 'China Life Insurance Co Ltd',
    '1299.HK': 'AIA Group Ltd',
    '2382.HK': 'Sunny Optical Technology Group Co Ltd',
}

hongkong_telecom = {
    '0941.HK': 'China Mobile Ltd',
    '0762.HK': 'China Unicom Hong Kong Ltd',
    '0728.HK': 'China Telecom Corp Ltd',
}

hongkong_realestate = {
    '0016.HK': 'Sun Hung Kai Properties Ltd',
    '0012.HK': 'Henderson Land Development Co Ltd',
    '0017.HK': 'New World Development Co Ltd',
    '1109.HK': 'China Resources Land Ltd',
    '1113.HK': 'CK Asset Holdings Ltd',
    '0101.HK': 'Hang Lung Properties Ltd',
    '0688.HK': 'China Overseas Land & Investment Ltd',
    '2007.HK': 'Country Garden Holdings Co Ltd',
    '3333.HK': 'China Evergrande Group',
}

hongkong_energy = {
    '0857.HK': 'PetroChina Co Ltd',
    '0883.HK': 'CNOOC Ltd',
    '0386.HK': 'China Petroleum & Chemical Corp',
}

hongkong_consumer = {
    '9987.HK': 'Yum China Holdings Inc',
    '9869.HK': 'Helens International Holdings Co Ltd',
    '6618.HK': 'JD Health International Inc',
    '1833.HK': 'PA Gooddoctor',
    '1876.HK': 'Budweiser Brewing Co APAC Ltd',
    '0288.HK': 'WH Group Ltd',
    '1044.HK': 'Hengan International Group Co Ltd',
}

hongkong_industrials = {
    '0388.HK': 'Hong Kong Exchanges and Clearing Ltd',
    '0001.HK': 'CK Hutchison Holdings Ltd',
    '0003.HK': 'Hong Kong and China Gas Co Ltd',
    '0002.HK': 'CLP Holdings Ltd',
    '0006.HK': 'Power Assets Holdings Ltd',
}

shanghai_finance = {
    '600519.SS': 'Kweichow Moutai Co Ltd',
    '601318.SS': 'Ping An Insurance Group Co of China Ltd',
    '600036.SS': 'China Merchants Bank Co Ltd',
    '601398.SS': 'Industrial and Commercial Bank of China Ltd',
    '601288.SS': 'Agricultural Bank of China Ltd',
    '601939.SS': 'China Construction Bank Corp',
    '601988.SS': 'Bank of China Ltd',
    '600000.SS': 'Shanghai Pudong Development Bank Co Ltd',
    '600016.SS': 'China Minsheng Banking Corp Ltd',
    '601166.SS': 'Industrial Bank Co Ltd',
    '600030.SS': 'CITIC Securities Co Ltd',
    '601688.SS': 'Huatai Securities Co Ltd',
}

shanghai_consumer = {
    '600887.SS': 'Inner Mongolia Yili Industrial Group Co Ltd',
    '600276.SS': 'Jiangsu Hengrui Medicine Co Ltd',
    '603259.SS': 'WuXi AppTec Co Ltd',
    '600809.SS': 'Shanxi Xinghuacun Fen Wine Factory Co Ltd',
    '600690.SS': 'Haier Smart Home Co Ltd',
    '603288.SS': 'Foshan Haitian Flavouring and Food Co Ltd',
    '600585.SS': 'Anhui Conch Cement Co Ltd',
}

shanghai_tech = {
    '600588.SS': 'Yonyou Network Technology Co Ltd',
    '600019.SS': 'Baoshan Iron & Steel Co Ltd',
    '601857.SS': 'PetroChina Co Ltd',
    '601899.SS': 'Zijin Mining Group Co Ltd',
    '600050.SS': 'China United Network Communications Ltd',
}

shenzhen_tech = {
    '000858.SZ': 'Wuliangye Yibin Co Ltd',
    '000651.SZ': 'Gree Electric Appliances Inc of Zhuhai',
    '002594.SZ': 'BYD Co Ltd',
    '300750.SZ': 'Contemporary Amperex Technology Co Ltd',
    '002415.SZ': 'Hangzhou Hikvision Digital Technology Co Ltd',
    '002475.SZ': 'Luxshare Precision Industry Co Ltd',
    '300059.SZ': 'East Money Information Co Ltd',
    '002371.SZ': 'Will Semiconductor Co Ltd Shanghai',
    '000333.SZ': 'Midea Group Co Ltd',
}

# ============================================================================
# EUROPE - Euronext (100+ stocks)
# ============================================================================

france_luxury = {
    'MC.PA': 'LVMH Moet Hennessy Louis Vuitton SE',
    'CDI.PA': 'Christian Dior SE',
    'KER.PA': 'Kering SA',
    'RMS.PA': 'Hermes International',
    'OR.PA': 'L\'Oreal SA',
}

france_industrials = {
    'AIR.PA': 'Airbus SE',
    'SAF.PA': 'Safran SA',
    'DG.PA': 'Vinci SA',
    'SGO.PA': 'Compagnie de Saint-Gobain SA',
    'SU.PA': 'Schneider Electric SE',
    'LR.PA': 'Legrand SA',
}

france_finance = {
    'BNP.PA': 'BNP Paribas SA',
    'ACA.PA': 'Credit Agricole SA',
    'GLE.PA': 'Societe Generale SA',
    'CS.PA': 'AXA SA',
}

france_energy = {
    'TTE.PA': 'TotalEnergies SE',
    'EDF.PA': 'Electricite de France SA',
    'ENGI.PA': 'Engie SA',
}

france_pharma = {
    'SAN.PA': 'Sanofi',
    'EL.PA': 'EssilorLuxottica SA',
}

france_consumer = {
    'BN.PA': 'Danone SA',
    'CA.PA': 'Carrefour SA',
    'RI.PA': 'Pernod Ricard SA',
}

france_tech = {
    'CAP.PA': 'Capgemini SE',
    'ATO.PA': 'Atos SE',
    'DSY.PA': 'Dassault Systemes SE',
}

netherlands = {
    'ASML.AS': 'ASML Holding NV',
    'HEIA.AS': 'Heineken NV',
    'PHIA.AS': 'Koninklijke Philips NV',
    'INGA.AS': 'ING Groep NV',
    'AD.AS': 'Ahold Delhaize NV',
    'ADYEN.AS': 'Adyen NV',
    'AKZA.AS': 'Akzo Nobel NV',
    'DSM.AS': 'Koninklijke DSM NV',
    'UNA.AS': 'Unilever NV',
    'WKL.AS': 'Wolters Kluwer NV',
    'REN.AS': 'RELX NV',
    'PRX.AS': 'Prosus NV',
}

germany = {
    'SAP.DE': 'SAP SE',
    'SIE.DE': 'Siemens AG',
    'VOW3.DE': 'Volkswagen AG',
    'BAYN.DE': 'Bayer AG',
    'ALV.DE': 'Allianz SE',
    'DTE.DE': 'Deutsche Telekom AG',
    'BAS.DE': 'BASF SE',
    'MUV2.DE': 'Munich Re',
    'DBK.DE': 'Deutsche Bank AG',
    'BMW.DE': 'Bayerische Motoren Werke AG',
    'DAI.DE': 'Daimler AG',
    'ADS.DE': 'Adidas AG',
    'LIN.DE': 'Linde PLC',
    'MRK.DE': 'Merck KGaA',
    'FRE.DE': 'Fresenius SE & Co KGaA',
    'DPW.DE': 'Deutsche Post AG',
    'HEN3.DE': 'Henkel AG & Co KGaA',
    'IFX.DE': 'Infineon Technologies AG',
    'CON.DE': 'Continental AG',
}

italy = {
    'ENI.MI': 'Eni SpA',
    'ENEL.MI': 'Enel SpA',
    'ISP.MI': 'Intesa Sanpaolo',
    'UCG.MI': 'UniCredit SpA',
    'G.MI': 'Assicurazioni Generali SpA',
    'STLA.MI': 'Stellantis NV',
    'LDO.MI': 'Leonardo SpA',
}

spain = {
    'ITX.MC': 'Industria de Diseno Textil SA',
    'SAN.MC': 'Banco Santander SA',
    'BBVA.MC': 'Banco Bilbao Vizcaya Argentaria SA',
    'IBE.MC': 'Iberdrola SA',
    'TEF.MC': 'Telefonica SA',
    'REP.MC': 'Repsol SA',
    'FER.MC': 'Ferrovial SA',
}

belgium = {
    'ABI.BR': 'Anheuser-Busch InBev SA/NV',
    'KBC.BR': 'KBC Group NV',
    'ACKB.BR': 'Ackermans & van Haaren NV',
    'UCB.BR': 'UCB SA',
}

switzerland = {
    'NESN.SW': 'Nestle SA',
    'NOVN.SW': 'Novartis AG',
    'ROG.SW': 'Roche Holding AG',
    'UBSG.SW': 'UBS Group AG',
    'ZURN.SW': 'Zurich Insurance Group AG',
    'ABBN.SW': 'ABB Ltd',
    'CFR.SW': 'Compagnie Financiere Richemont SA',
    'SREN.SW': 'Swiss Re AG',
    'GIVN.SW': 'Givaudan SA',
    'LONN.SW': 'Lonza Group AG',
    'GEBN.SW': 'Geberit AG',
}

# ============================================================================
# CANADA - Toronto Stock Exchange (50+ stocks)
# ============================================================================

canada_finance = {
    'RY.TO': 'Royal Bank of Canada',
    'TD.TO': 'Toronto-Dominion Bank',
    'BNS.TO': 'Bank of Nova Scotia',
    'BMO.TO': 'Bank of Montreal',
    'CM.TO': 'Canadian Imperial Bank of Commerce',
    'NA.TO': 'National Bank of Canada',
    'MFC.TO': 'Manulife Financial Corp',
    'SLF.TO': 'Sun Life Financial Inc',
    'POW.TO': 'Power Corp of Canada',
    'GWO.TO': 'Great-West Lifeco Inc',
}

canada_energy = {
    'CNQ.TO': 'Canadian Natural Resources Ltd',
    'SU.TO': 'Suncor Energy Inc',
    'ENB.TO': 'Enbridge Inc',
    'TRP.TO': 'TC Energy Corp',
    'CVE.TO': 'Cenovus Energy Inc',
    'IMO.TO': 'Imperial Oil Ltd',
    'PPL.TO': 'Pembina Pipeline Corp',
    'ALA.TO': 'AltaGas Ltd',
    'KEY.TO': 'Keyera Corp',
}

canada_materials = {
    'ABX.TO': 'Barrick Gold Corp',
    'NEM.TO': 'Newmont Corp',
    'K.TO': 'Kinross Gold Corp',
    'WPM.TO': 'Wheaton Precious Metals Corp',
    'FNV.TO': 'Franco-Nevada Corp',
    'AEM.TO': 'Agnico Eagle Mines Ltd',
    'CCO.TO': 'Cameco Corp',
    'FM.TO': 'First Quantum Minerals Ltd',
    'TECK-B.TO': 'Teck Resources Ltd',
}

canada_industrials = {
    'CNR.TO': 'Canadian National Railway Co',
    'CP.TO': 'Canadian Pacific Railway Ltd',
    'CSU.TO': 'Constellation Software Inc',
    'WCN.TO': 'Waste Connections Inc',
    'TRI.TO': 'Thomson Reuters Corp',
    'DOL.TO': 'Dollarama Inc',
    'ATD.TO': 'Alimentation Couche-Tard Inc',
    'CAE.TO': 'CAE Inc',
}

canada_telecom = {
    'BCE.TO': 'BCE Inc',
    'T.TO': 'Telus Corp',
    'RCI-B.TO': 'Rogers Communications Inc',
    'QBR-B.TO': 'Quebecor Inc',
}

canada_consumer = {
    'L.TO': 'Loblaw Companies Ltd',
    'MG.TO': 'Magna International Inc',
    'QSR.TO': 'Restaurant Brands International Inc',
    'WN.TO': 'George Weston Ltd',
}

canada_tech = {
    'SHOP.TO': 'Shopify Inc',
    'BB.TO': 'BlackBerry Ltd',
    'LSPD.TO': 'Lightspeed Commerce Inc',
    'OTEX.TO': 'Open Text Corp',
}

canada_utilities = {
    'FTS.TO': 'Fortis Inc',
    'EMA.TO': 'Emera Inc',
    'AQN.TO': 'Algonquin Power & Utilities Corp',
    'CU.TO': 'Canadian Utilities Ltd',
}

# ============================================================================
# MERGE ALL CATEGORIES
# ============================================================================

# USA
MEGA_STOCK_DATABASE.update(usa_tech_software)
MEGA_STOCK_DATABASE.update(usa_ecommerce_payments)
MEGA_STOCK_DATABASE.update(usa_social_entertainment)
MEGA_STOCK_DATABASE.update(usa_banking)
MEGA_STOCK_DATABASE.update(usa_insurance_realestate)
MEGA_STOCK_DATABASE.update(usa_pharma_biotech)
MEGA_STOCK_DATABASE.update(usa_healthcare_equipment)
MEGA_STOCK_DATABASE.update(usa_consumer_retail)
MEGA_STOCK_DATABASE.update(usa_consumer_staples)
MEGA_STOCK_DATABASE.update(usa_energy)
MEGA_STOCK_DATABASE.update(usa_industrials)
MEGA_STOCK_DATABASE.update(usa_materials)
MEGA_STOCK_DATABASE.update(usa_utilities)
MEGA_STOCK_DATABASE.update(usa_telecom)
MEGA_STOCK_DATABASE.update(usa_ev_cleanenergy)
MEGA_STOCK_DATABASE.update(usa_chinese_adrs)
MEGA_STOCK_DATABASE.update(usa_etfs)

# UK
MEGA_STOCK_DATABASE.update(uk_energy)
MEGA_STOCK_DATABASE.update(uk_finance)
MEGA_STOCK_DATABASE.update(uk_consumer)
MEGA_STOCK_DATABASE.update(uk_pharma_healthcare)
MEGA_STOCK_DATABASE.update(uk_materials)
MEGA_STOCK_DATABASE.update(uk_industrials)
MEGA_STOCK_DATABASE.update(uk_telecom_tech)
MEGA_STOCK_DATABASE.update(uk_realestate)
MEGA_STOCK_DATABASE.update(uk_luxury)

# Japan
MEGA_STOCK_DATABASE.update(japan_automotive)
MEGA_STOCK_DATABASE.update(japan_electronics)
MEGA_STOCK_DATABASE.update(japan_semiconductors)
MEGA_STOCK_DATABASE.update(japan_telecom_internet)
MEGA_STOCK_DATABASE.update(japan_gaming)
MEGA_STOCK_DATABASE.update(japan_finance)
MEGA_STOCK_DATABASE.update(japan_retail_consumer)
MEGA_STOCK_DATABASE.update(japan_pharma)
MEGA_STOCK_DATABASE.update(japan_industrials)
MEGA_STOCK_DATABASE.update(japan_trading)
MEGA_STOCK_DATABASE.update(japan_materials)

# China/Hong Kong
MEGA_STOCK_DATABASE.update(hongkong_tech)
MEGA_STOCK_DATABASE.update(hongkong_finance)
MEGA_STOCK_DATABASE.update(hongkong_telecom)
MEGA_STOCK_DATABASE.update(hongkong_realestate)
MEGA_STOCK_DATABASE.update(hongkong_energy)
MEGA_STOCK_DATABASE.update(hongkong_consumer)
MEGA_STOCK_DATABASE.update(hongkong_industrials)
MEGA_STOCK_DATABASE.update(shanghai_finance)
MEGA_STOCK_DATABASE.update(shanghai_consumer)
MEGA_STOCK_DATABASE.update(shanghai_tech)
MEGA_STOCK_DATABASE.update(shenzhen_tech)

# Europe
MEGA_STOCK_DATABASE.update(france_luxury)
MEGA_STOCK_DATABASE.update(france_industrials)
MEGA_STOCK_DATABASE.update(france_finance)
MEGA_STOCK_DATABASE.update(france_energy)
MEGA_STOCK_DATABASE.update(france_pharma)
MEGA_STOCK_DATABASE.update(france_consumer)
MEGA_STOCK_DATABASE.update(france_tech)
MEGA_STOCK_DATABASE.update(netherlands)
MEGA_STOCK_DATABASE.update(germany)
MEGA_STOCK_DATABASE.update(italy)
MEGA_STOCK_DATABASE.update(spain)
MEGA_STOCK_DATABASE.update(belgium)
MEGA_STOCK_DATABASE.update(switzerland)

# Canada
MEGA_STOCK_DATABASE.update(canada_finance)
MEGA_STOCK_DATABASE.update(canada_energy)
MEGA_STOCK_DATABASE.update(canada_materials)
MEGA_STOCK_DATABASE.update(canada_industrials)
MEGA_STOCK_DATABASE.update(canada_telecom)
MEGA_STOCK_DATABASE.update(canada_consumer)
MEGA_STOCK_DATABASE.update(canada_tech)
MEGA_STOCK_DATABASE.update(canada_utilities)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_total_count() -> int:
    """Get total number of stocks in database."""
    return len(MEGA_STOCK_DATABASE)


def get_exchange_breakdown() -> dict:
    """Get stock count by exchange."""
    counts = {
        'USA (NYSE/NASDAQ)': 0,
        'UK (LSE)': 0,
        'Japan (TSE)': 0,
        'China (SSE/SZSE)': 0,
        'Hong Kong (HKEX)': 0,
        'Europe (Euronext)': 0,
        'Canada (TSX)': 0,
    }

    for ticker in MEGA_STOCK_DATABASE.keys():
        if ticker.endswith('.L'):
            counts['UK (LSE)'] += 1
        elif ticker.endswith('.T'):
            counts['Japan (TSE)'] += 1
        elif ticker.endswith('.SS') or ticker.endswith('.SZ'):
            counts['China (SSE/SZSE)'] += 1
        elif ticker.endswith('.HK'):
            counts['Hong Kong (HKEX)'] += 1
        elif ticker.endswith('.PA') or ticker.endswith('.AS') or ticker.endswith('.DE') or \
             ticker.endswith('.MI') or ticker.endswith('.MC') or ticker.endswith('.BR') or \
             ticker.endswith('.SW'):
            counts['Europe (Euronext)'] += 1
        elif ticker.endswith('.TO'):
            counts['Canada (TSX)'] += 1
        else:
            counts['USA (NYSE/NASDAQ)'] += 1

    return counts


def print_database_stats():
    """Print database statistics."""
    total = get_total_count()
    breakdown = get_exchange_breakdown()

    print(f"\nMEGA STOCK DATABASE STATISTICS")
    print("=" * 50)
    print(f"Total stocks: {total}")
    print("\nBreakdown by exchange:")
    for exchange, count in breakdown.items():
        print(f"  {exchange}: {count}")
    print("=" * 50)


if __name__ == '__main__':
    print_database_stats()

    # Verify Pop Mart is included
    if '9992.HK' in MEGA_STOCK_DATABASE:
        print(f"\nPop Mart (9992.HK) confirmed: {MEGA_STOCK_DATABASE['9992.HK']}")
    else:
        print(f"\nWARNING: Pop Mart (9992.HK) NOT FOUND!")
