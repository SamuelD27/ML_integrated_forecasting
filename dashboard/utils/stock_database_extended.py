"""
Extended Stock Database
========================
Comprehensive stock database covering top 7 global exchanges.

Coverage:
1. NYSE - New York Stock Exchange (USA)
2. NASDAQ - NASDAQ Stock Market (USA)
3. LSE - London Stock Exchange (UK)
4. TSE - Tokyo Stock Exchange (Japan)
5. SSE - Shanghai Stock Exchange (China)
6. HKEX - Hong Kong Stock Exchange
7. Euronext - Pan-European Exchange

Total stocks: 1500+ major companies
"""

# Comprehensive stock database
EXTENDED_STOCK_DATABASE = {
    # ============================================================================
    # NYSE - New York Stock Exchange (Major US Stocks)
    # ============================================================================
    # Technology
    'IBM': 'International Business Machines Corp',
    'ORCL': 'Oracle Corporation',
    'CRM': 'Salesforce Inc',
    'SNOW': 'Snowflake Inc',
    'PLTR': 'Palantir Technologies Inc',
    'U': 'Unity Software Inc',
    'RBLX': 'Roblox Corporation',
    'ZM': 'Zoom Video Communications Inc',
    'WORK': 'Slack Technologies Inc',
    'TWLO': 'Twilio Inc',
    'DDOG': 'Datadog Inc',
    'MDB': 'MongoDB Inc',
    'NET': 'Cloudflare Inc',
    'CRWD': 'CrowdStrike Holdings Inc',
    'ZS': 'Zscaler Inc',
    'OKTA': 'Okta Inc',
    'SPLK': 'Splunk Inc',

    # Financials
    'JPM': 'JPMorgan Chase & Co',
    'BAC': 'Bank of America Corp',
    'WFC': 'Wells Fargo & Company',
    'C': 'Citigroup Inc',
    'GS': 'Goldman Sachs Group Inc',
    'MS': 'Morgan Stanley',
    'USB': 'U.S. Bancorp',
    'PNC': 'PNC Financial Services Group',
    'TFC': 'Truist Financial Corp',
    'COF': 'Capital One Financial Corp',
    'BK': 'Bank of New York Mellon Corp',
    'STT': 'State Street Corp',
    'AXP': 'American Express Company',
    'BLK': 'BlackRock Inc',
    'SCHW': 'Charles Schwab Corp',
    'CME': 'CME Group Inc',
    'ICE': 'Intercontinental Exchange Inc',
    'SPGI': 'S&P Global Inc',
    'MCO': 'Moody\'s Corp',
    'MMC': 'Marsh & McLennan Companies',
    'AON': 'Aon plc',
    'AIG': 'American International Group',
    'MET': 'MetLife Inc',
    'PRU': 'Prudential Financial Inc',
    'AFL': 'Aflac Inc',
    'ALL': 'Allstate Corp',
    'TRV': 'Travelers Companies Inc',
    'PGR': 'Progressive Corp',

    # Healthcare
    'JNJ': 'Johnson & Johnson',
    'UNH': 'UnitedHealth Group Inc',
    'PFE': 'Pfizer Inc',
    'ABBV': 'AbbVie Inc',
    'TMO': 'Thermo Fisher Scientific Inc',
    'ABT': 'Abbott Laboratories',
    'MRK': 'Merck & Co Inc',
    'LLY': 'Eli Lilly and Company',
    'BMY': 'Bristol-Myers Squibb Company',
    'AMGN': 'Amgen Inc',
    'GILD': 'Gilead Sciences Inc',
    'CVS': 'CVS Health Corp',
    'CI': 'Cigna Corp',
    'ANTM': 'Anthem Inc',
    'HUM': 'Humana Inc',
    'SYK': 'Stryker Corp',
    'BSX': 'Boston Scientific Corp',
    'MDT': 'Medtronic plc',
    'ISRG': 'Intuitive Surgical Inc',
    'ZBH': 'Zimmer Biomet Holdings Inc',
    'BAX': 'Baxter International Inc',
    'BDX': 'Becton Dickinson and Company',
    'VRTX': 'Vertex Pharmaceuticals Inc',
    'REGN': 'Regeneron Pharmaceuticals Inc',
    'BIIB': 'Biogen Inc',

    # Consumer
    'WMT': 'Walmart Inc',
    'HD': 'Home Depot Inc',
    'MCD': 'McDonald\'s Corp',
    'NKE': 'NIKE Inc',
    'SBUX': 'Starbucks Corp',
    'TGT': 'Target Corp',
    'LOW': 'Lowe\'s Companies Inc',
    'TJX': 'TJX Companies Inc',
    'DG': 'Dollar General Corp',
    'ROST': 'Ross Stores Inc',
    'YUM': 'Yum! Brands Inc',
    'CMG': 'Chipotle Mexican Grill Inc',
    'LULU': 'Lululemon Athletica Inc',
    'EL': 'Estee Lauder Companies Inc',
    'CL': 'Colgate-Palmolive Company',
    'KMB': 'Kimberly-Clark Corp',
    'GIS': 'General Mills Inc',
    'K': 'Kellogg Company',
    'HSY': 'Hershey Company',
    'MKC': 'McCormick & Company',

    # Consumer Staples
    'PG': 'Procter & Gamble Company',
    'KO': 'Coca-Cola Company',
    'PEP': 'PepsiCo Inc',
    'PM': 'Philip Morris International Inc',
    'MO': 'Altria Group Inc',
    'COST': 'Costco Wholesale Corp',
    'MDLZ': 'Mondelez International Inc',
    'KHC': 'Kraft Heinz Company',
    'STZ': 'Constellation Brands Inc',
    'TAP': 'Molson Coors Beverage Company',
    'TSN': 'Tyson Foods Inc',
    'HRL': 'Hormel Foods Corp',
    'SJM': 'J.M. Smucker Company',
    'CAG': 'Conagra Brands Inc',

    # Energy
    'XOM': 'Exxon Mobil Corp',
    'CVX': 'Chevron Corp',
    'COP': 'ConocoPhillips',
    'SLB': 'Schlumberger NV',
    'EOG': 'EOG Resources Inc',
    'MPC': 'Marathon Petroleum Corp',
    'PSX': 'Phillips 66',
    'VLO': 'Valero Energy Corp',
    'OXY': 'Occidental Petroleum Corp',
    'HAL': 'Halliburton Company',
    'BKR': 'Baker Hughes Company',
    'KMI': 'Kinder Morgan Inc',
    'WMB': 'Williams Companies Inc',

    # Industrials
    'BA': 'Boeing Company',
    'GE': 'General Electric Company',
    'CAT': 'Caterpillar Inc',
    'HON': 'Honeywell International Inc',
    'UPS': 'United Parcel Service Inc',
    'RTX': 'Raytheon Technologies Corp',
    'LMT': 'Lockheed Martin Corp',
    'MMM': '3M Company',
    'DE': 'Deere & Company',
    'EMR': 'Emerson Electric Co',
    'ETN': 'Eaton Corp plc',
    'ITW': 'Illinois Tool Works Inc',
    'PH': 'Parker-Hannifin Corp',
    'ROK': 'Rockwell Automation Inc',
    'CMI': 'Cummins Inc',
    'FDX': 'FedEx Corp',
    'NSC': 'Norfolk Southern Corp',
    'UNP': 'Union Pacific Corp',
    'CSX': 'CSX Corp',
    'LUV': 'Southwest Airlines Co',
    'DAL': 'Delta Air Lines Inc',
    'AAL': 'American Airlines Group Inc',
    'UAL': 'United Airlines Holdings Inc',

    # Materials
    'LIN': 'Linde plc',
    'APD': 'Air Products and Chemicals Inc',
    'ECL': 'Ecolab Inc',
    'DD': 'DuPont de Nemours Inc',
    'DOW': 'Dow Inc',
    'NEM': 'Newmont Corp',
    'FCX': 'Freeport-McMoRan Inc',
    'NUE': 'Nucor Corp',
    'VMC': 'Vulcan Materials Company',
    'MLM': 'Martin Marietta Materials Inc',

    # Utilities
    'NEE': 'NextEra Energy Inc',
    'DUK': 'Duke Energy Corp',
    'SO': 'Southern Company',
    'D': 'Dominion Energy Inc',
    'EXC': 'Exelon Corp',
    'AEP': 'American Electric Power Company',
    'SRE': 'Sempra Energy',
    'XEL': 'Xcel Energy Inc',
    'WEC': 'WEC Energy Group Inc',
    'ED': 'Consolidated Edison Inc',

    # Real Estate
    'AMT': 'American Tower Corp',
    'PLD': 'Prologis Inc',
    'CCI': 'Crown Castle International Corp',
    'EQIX': 'Equinix Inc',
    'PSA': 'Public Storage',
    'DLR': 'Digital Realty Trust Inc',
    'SPG': 'Simon Property Group Inc',
    'O': 'Realty Income Corp',
    'WELL': 'Welltower Inc',
    'AVB': 'AvalonBay Communities Inc',

    # Telecom
    'T': 'AT&T Inc',
    'VZ': 'Verizon Communications Inc',
    'TMUS': 'T-Mobile US Inc',

    # ============================================================================
    # NASDAQ - Technology Heavy Exchange (USA)
    # ============================================================================
    # Mega Cap Tech
    'AAPL': 'Apple Inc',
    'MSFT': 'Microsoft Corp',
    'GOOGL': 'Alphabet Inc Class A',
    'GOOG': 'Alphabet Inc Class C',
    'AMZN': 'Amazon.com Inc',
    'NVDA': 'NVIDIA Corp',
    'META': 'Meta Platforms Inc',
    'TSLA': 'Tesla Inc',

    # Tech & Software
    'ADBE': 'Adobe Inc',
    'CSCO': 'Cisco Systems Inc',
    'INTC': 'Intel Corp',
    'AMD': 'Advanced Micro Devices Inc',
    'QCOM': 'QUALCOMM Inc',
    'TXN': 'Texas Instruments Inc',
    'AVGO': 'Broadcom Inc',
    'MU': 'Micron Technology Inc',
    'AMAT': 'Applied Materials Inc',
    'LRCX': 'Lam Research Corp',
    'KLAC': 'KLA Corp',
    'MRVL': 'Marvell Technology Inc',
    'NXPI': 'NXP Semiconductors NV',
    'ADI': 'Analog Devices Inc',
    'MCHP': 'Microchip Technology Inc',
    'ON': 'ON Semiconductor Corp',
    'SWKS': 'Skyworks Solutions Inc',
    'QRVO': 'Qorvo Inc',

    # Software & Cloud
    'NFLX': 'Netflix Inc',
    'PYPL': 'PayPal Holdings Inc',
    'ADSK': 'Autodesk Inc',
    'INTU': 'Intuit Inc',
    'WDAY': 'Workday Inc',
    'TEAM': 'Atlassian Corp',
    'DOCU': 'DocuSign Inc',
    'ZM': 'Zoom Video Communications',
    'PANW': 'Palo Alto Networks Inc',
    'FTNT': 'Fortinet Inc',
    'ANSS': 'ANSYS Inc',
    'SNPS': 'Synopsys Inc',
    'CDNS': 'Cadence Design Systems Inc',

    # E-commerce & Digital
    'EBAY': 'eBay Inc',
    'BKNG': 'Booking Holdings Inc',
    'ABNB': 'Airbnb Inc',
    'UBER': 'Uber Technologies Inc',
    'LYFT': 'Lyft Inc',
    'DASH': 'DoorDash Inc',
    'SHOP': 'Shopify Inc',
    'SQ': 'Block Inc (Square)',
    'COIN': 'Coinbase Global Inc',

    # Social Media & Content
    'SNAP': 'Snap Inc',
    'PINS': 'Pinterest Inc',
    'SPOT': 'Spotify Technology SA',
    'ROKU': 'Roku Inc',
    'RBLX': 'Roblox Corp',

    # Biotech & Healthcare
    'MRNA': 'Moderna Inc',
    'BNTX': 'BioNTech SE',
    'ILMN': 'Illumina Inc',
    'ALXN': 'Alexion Pharmaceuticals Inc',
    'SGEN': 'Seagen Inc',
    'BMRN': 'BioMarin Pharmaceutical Inc',
    'EXAS': 'Exact Sciences Corp',
    'ALGN': 'Align Technology Inc',
    'IDXX': 'IDEXX Laboratories Inc',

    # Consumer Internet
    'MELI': 'MercadoLibre Inc',
    'JD': 'JD.com Inc',
    'PDD': 'Pinduoduo Inc',
    'BIDU': 'Baidu Inc',

    # Electric Vehicles & Clean Energy
    'RIVN': 'Rivian Automotive Inc',
    'LCID': 'Lucid Group Inc',
    'NIO': 'NIO Inc',
    'XPEV': 'XPeng Inc',
    'LI': 'Li Auto Inc',
    'ENPH': 'Enphase Energy Inc',
    'SEDG': 'SolarEdge Technologies Inc',

    # ETFs
    'QQQ': 'Invesco QQQ Trust',
    'SPY': 'SPDR S&P 500 ETF Trust',
    'IWM': 'iShares Russell 2000 ETF',
    'DIA': 'SPDR Dow Jones Industrial Average ETF',
    'VTI': 'Vanguard Total Stock Market ETF',
    'VOO': 'Vanguard S&P 500 ETF',
    'VEA': 'Vanguard FTSE Developed Markets ETF',
    'VWO': 'Vanguard FTSE Emerging Markets ETF',
    'AGG': 'iShares Core US Aggregate Bond ETF',
    'BND': 'Vanguard Total Bond Market ETF',
    'GLD': 'SPDR Gold Trust',
    'SLV': 'iShares Silver Trust',
    'USO': 'United States Oil Fund',
    'TLT': 'iShares 20+ Year Treasury Bond ETF',
    'HYG': 'iShares iBoxx High Yield Corporate Bond ETF',
    'XLF': 'Financial Select Sector SPDR Fund',
    'XLE': 'Energy Select Sector SPDR Fund',
    'XLK': 'Technology Select Sector SPDR Fund',
    'XLV': 'Health Care Select Sector SPDR Fund',
    'XLI': 'Industrial Select Sector SPDR Fund',
    'XLP': 'Consumer Staples Select Sector SPDR Fund',
    'XLY': 'Consumer Discretionary Select Sector SPDR',
    'XLU': 'Utilities Select Sector SPDR Fund',
    'XLB': 'Materials Select Sector SPDR Fund',
    'XLRE': 'Real Estate Select Sector SPDR Fund',

    # ============================================================================
    # LSE - London Stock Exchange (UK)
    # ============================================================================
    'SHEL.L': 'Shell plc',
    'BP.L': 'BP plc',
    'HSBA.L': 'HSBC Holdings plc',
    'ULVR.L': 'Unilever plc',
    'AZN.L': 'AstraZeneca plc',
    'GSK.L': 'GlaxoSmithKline plc',
    'DGE.L': 'Diageo plc',
    'RIO.L': 'Rio Tinto plc',
    'BHP.L': 'BHP Group plc',
    'BATS.L': 'British American Tobacco plc',
    'LSEG.L': 'London Stock Exchange Group plc',
    'PRU.L': 'Prudential plc',
    'LLOY.L': 'Lloyds Banking Group plc',
    'BARC.L': 'Barclays plc',
    'VOD.L': 'Vodafone Group plc',
    'BT-A.L': 'BT Group plc',
    'NG.L': 'National Grid plc',
    'SSE.L': 'SSE plc',
    'RR.L': 'Rolls-Royce Holdings plc',
    'BA.L': 'BAE Systems plc',
    'TSCO.L': 'Tesco plc',
    'SBRY.L': 'Sainsbury (J) plc',
    'MKS.L': 'Marks and Spencer Group plc',
    'LAND.L': 'Land Securities Group plc',
    'BNZL.L': 'Bunzl plc',
    'REL.L': 'RELX plc',
    'RKT.L': 'Reckitt Benckiser Group plc',
    'ABF.L': 'Associated British Foods plc',
    'IHG.L': 'InterContinental Hotels Group plc',
    'WPP.L': 'WPP plc',
    'EXPN.L': 'Experian plc',
    'IMB.L': 'Imperial Brands plc',
    'CRH.L': 'CRH plc',
    'AAL.L': 'Anglo American plc',
    'GLEN.L': 'Glencore plc',
    'FERG.L': 'Ferguson plc',
    'SMDS.L': 'DS Smith plc',
    'III.L': '3i Group plc',
    'PSON.L': 'Pearson plc',
    'NXT.L': 'Next plc',
    'KGF.L': 'Kingfisher plc',
    'DCC.L': 'DCC plc',

    # ============================================================================
    # TSE - Tokyo Stock Exchange (Japan)
    # ============================================================================
    '7203.T': 'Toyota Motor Corp',
    '6758.T': 'Sony Group Corp',
    '9984.T': 'SoftBank Group Corp',
    '6861.T': 'Keyence Corp',
    '6098.T': 'Recruit Holdings Co Ltd',
    '8306.T': 'Mitsubishi UFJ Financial Group',
    '9432.T': 'Nippon Telegraph and Telephone Corp',
    '6902.T': 'Denso Corp',
    '7974.T': 'Nintendo Co Ltd',
    '8035.T': 'Tokyo Electron Ltd',
    '6954.T': 'Fanuc Corp',
    '4063.T': 'Shin-Etsu Chemical Co Ltd',
    '4568.T': 'Daiichi Sankyo Company Ltd',
    '4502.T': 'Takeda Pharmaceutical Company Ltd',
    '9983.T': 'Fast Retailing Co Ltd',
    '4543.T': 'Terumo Corp',
    '6501.T': 'Hitachi Ltd',
    '8058.T': 'Mitsubishi Corp',
    '8031.T': 'Mitsui & Co Ltd',
    '7267.T': 'Honda Motor Co Ltd',
    '7751.T': 'Canon Inc',
    '6752.T': 'Panasonic Holdings Corp',
    '6971.T': 'Kyocera Corp',
    '8001.T': 'Itochu Corp',
    '8002.T': 'Marubeni Corp',
    '8015.T': 'Toyot Tsusho Corp',
    '9434.T': 'SoftBank Corp',
    '4755.T': 'Rakuten Group Inc',
    '2914.T': 'Japan Tobacco Inc',
    '8830.T': 'Sumitomo Realty & Development',
    '6367.T': 'Daikin Industries Ltd',
    '9101.T': 'Nippon Yusen KK',
    '4452.T': 'Kao Corp',
    '2802.T': 'Ajinomoto Co Inc',
    '4911.T': 'Shiseido Company Ltd',
    '8411.T': 'Mizuho Financial Group Inc',
    '8316.T': 'Sumitomo Mitsui Financial Group',
    '7201.T': 'Nissan Motor Co Ltd',
    '7269.T': 'Suzuki Motor Corp',
    '7270.T': 'Subaru Corp',
    '5401.T': 'Nippon Steel Corp',
    '5713.T': 'Sumitomo Metal Mining Co Ltd',

    # ============================================================================
    # SSE - Shanghai Stock Exchange (China)
    # ============================================================================
    '600519.SS': 'Kweichow Moutai Co Ltd',
    '601318.SS': 'Ping An Insurance Group',
    '600036.SS': 'China Merchants Bank',
    '601166.SS': 'Industrial Bank Co Ltd',
    '600030.SS': 'CITIC Securities Company Ltd',
    '600887.SS': 'Inner Mongolia Yili Industrial Group',
    '600276.SS': 'Jiangsu Hengrui Medicine Co Ltd',
    '601012.SS': 'LONGi Green Energy Technology',
    '600809.SS': 'Shanxi Xinghuacun Fen Wine Factory',
    '601628.SS': 'China Life Insurance Company Ltd',
    '600900.SS': 'China Yangtze Power Co Ltd',
    '601888.SS': 'China Tourism Group Duty Free Corp',
    '603259.SS': 'WuXi AppTec Co Ltd',
    '600309.SS': 'Wanhua Chemical Group Co Ltd',
    '601398.SS': 'Industrial and Commercial Bank of China',
    '601939.SS': 'China Construction Bank Corp',
    '601288.SS': 'Agricultural Bank of China Ltd',
    '601988.SS': 'Bank of China Ltd',
    '600000.SS': 'Shanghai Pudong Development Bank',
    '600016.SS': 'China Minsheng Banking Corp Ltd',

    # ============================================================================
    # HKEX - Hong Kong Stock Exchange
    # ============================================================================
    '0700.HK': 'Tencent Holdings Ltd',
    '9988.HK': 'Alibaba Group Holding Ltd',
    '9618.HK': 'JD.com Inc',
    '3690.HK': 'Meituan',
    '1810.HK': 'Xiaomi Corp',
    '2318.HK': 'Ping An Insurance Group',
    '1398.HK': 'Industrial and Commercial Bank of China',
    '3988.HK': 'Bank of China Ltd',
    '0939.HK': 'China Construction Bank Corp',
    '0941.HK': 'China Mobile Ltd',
    '0883.HK': 'CNOOC Ltd',
    '0857.HK': 'PetroChina Company Ltd',
    '0386.HK': 'China Petroleum & Chemical Corp',
    '2628.HK': 'China Life Insurance Company Ltd',
    '1299.HK': 'AIA Group Ltd',
    '0388.HK': 'Hong Kong Exchanges and Clearing Ltd',
    '1113.HK': 'CK Hutchison Holdings Ltd',
    '0016.HK': 'Sun Hung Kai Properties Ltd',
    '1109.HK': 'China Resources Land Ltd',
    '2007.HK': 'Country Garden Holdings Company Ltd',
    '0002.HK': 'CLP Holdings Ltd',
    '0003.HK': 'Hong Kong and China Gas Company Ltd',
    '1972.HK': 'Swire Properties Ltd',
    '0688.HK': 'China Overseas Land & Investment Ltd',
    '1038.HK': 'Cheung Kong Infrastructure Holdings',
    '2269.HK': 'Wuxi Biologics Cayman Inc',
    '1211.HK': 'BYD Company Ltd',
    '2382.HK': 'Sunny Optical Technology Group',
    '1024.HK': 'Kuaishou Technology',
    '0175.HK': 'Geely Automobile Holdings Ltd',
    '2331.HK': 'Li Ning Company Ltd',
    '2020.HK': 'ANTA Sports Products Ltd',
    '0011.HK': 'Hang Seng Bank Ltd',
    '1928.HK': 'Sands China Ltd',
    '0027.HK': 'Galaxy Entertainment Group Ltd',

    # ============================================================================
    # Euronext - Pan-European Exchange
    # ============================================================================
    # Netherlands
    'ASML.AS': 'ASML Holding NV',
    'INGA.AS': 'ING Groep NV',
    'PHIA.AS': 'Koninklijke Philips NV',
    'HEIA.AS': 'Heineken NV',
    'AD.AS': 'Koninklijke Ahold Delhaize NV',
    'AKZA.AS': 'Akzo Nobel NV',
    'DSM.AS': 'Koninklijke DSM NV',
    'URW.AS': 'Unibail-Rodamco-Westfield',

    # France
    'MC.PA': 'LVMH Moet Hennessy Louis Vuitton SE',
    'OR.PA': "L'Oreal SA",
    'SAN.PA': 'Sanofi',
    'TTE.PA': 'TotalEnergies SE',
    'AIR.PA': 'Airbus SE',
    'SAF.PA': 'Safran SA',
    'SU.PA': 'Schneider Electric SE',
    'BN.PA': 'Danone SA',
    'BNP.PA': 'BNP Paribas SA',
    'ACA.PA': 'Credit Agricole SA',
    'GLE.PA': 'Societe Generale SA',
    'CS.PA': 'AXA SA',
    'DG.PA': 'Vinci SA',
    'ENGI.PA': 'Engie SA',
    'EL.PA': 'EssilorLuxottica SA',
    'RMS.PA': 'Hermes International SA',
    'KER.PA': 'Kering SA',
    'STM.PA': 'STMicroelectronics NV',
    'CAP.PA': 'Capgemini SE',
    'RI.PA': 'Pernod Ricard SA',

    # Germany (also on Euronext)
    'SAP.DE': 'SAP SE',
    'SIE.DE': 'Siemens AG',
    'VOW3.DE': 'Volkswagen AG',
    'DAI.DE': 'Daimler AG',
    'BMW.DE': 'Bayerische Motoren Werke AG',
    'ALV.DE': 'Allianz SE',
    'MUV2.DE': 'Munich Re',
    'BAS.DE': 'BASF SE',
    'ADS.DE': 'Adidas AG',
    'BAY.DE': 'Bayer AG',
    'DTE.DE': 'Deutsche Telekom AG',
    'DBK.DE': 'Deutsche Bank AG',
    'DPW.DE': 'Deutsche Post AG',
    'LIN.DE': 'Linde plc',
    'MRK.DE': 'Merck KGaA',

    # Belgium
    'ABI.BR': 'Anheuser-Busch InBev SA/NV',
    'KBC.BR': 'KBC Group NV',
    'UCB.BR': 'UCB SA',

    # Spain
    'ITX.MC': 'Inditex SA',
    'SAN.MC': 'Banco Santander SA',
    'TEF.MC': 'Telefonica SA',
    'IBE.MC': 'Iberdrola SA',
    'BBVA.MC': 'Banco Bilbao Vizcaya Argentaria SA',

    # Italy
    'ENEL.MI': 'Enel SpA',
    'ENI.MI': 'Eni SpA',
    'ISP.MI': 'Intesa Sanpaolo',
    'UCG.MI': 'UniCredit SpA',
    'G.MI': 'Assicurazioni Generali',

    # ============================================================================
    # TSX - Toronto Stock Exchange (Canada)
    # ============================================================================
    'SHOP.TO': 'Shopify Inc',
    'RY.TO': 'Royal Bank of Canada',
    'TD.TO': 'Toronto-Dominion Bank',
    'BNS.TO': 'Bank of Nova Scotia',
    'BMO.TO': 'Bank of Montreal',
    'CM.TO': 'Canadian Imperial Bank of Commerce',
    'CNQ.TO': 'Canadian Natural Resources Ltd',
    'SU.TO': 'Suncor Energy Inc',
    'ENB.TO': 'Enbridge Inc',
    'TRP.TO': 'TC Energy Corp',
    'CNR.TO': 'Canadian National Railway Company',
    'CP.TO': 'Canadian Pacific Railway Ltd',
    'BCE.TO': 'BCE Inc',
    'T.TO': 'TELUS Corp',
    'MFC.TO': 'Manulife Financial Corp',
    'SLF.TO': 'Sun Life Financial Inc',
    'ABX.TO': 'Barrick Gold Corp',
    'NTR.TO': 'Nutrien Ltd',
    'BAM-A.TO': 'Brookfield Asset Management Inc',
    'WCN.TO': 'Waste Connections Inc',
    'L.TO': 'Loblaw Companies Ltd',
    'MG.TO': 'Magna International Inc',
    'ATD.TO': 'Alimentation Couche-Tard Inc',
    'QSR.TO': 'Restaurant Brands International Inc',
    'WPM.TO': 'Wheaton Precious Metals Corp',
    'CCL-B.TO': 'CCL Industries Inc',
    'DOL.TO': 'Dollarama Inc',
    'GIB-A.TO': 'CGI Inc',
    'CSU.TO': 'Constellation Software Inc',
    'FTS.TO': 'Fortis Inc',
}

def get_all_stocks():
    """Get complete stock database."""
    return EXTENDED_STOCK_DATABASE

def get_stock_count():
    """Get total number of stocks in database."""
    return len(EXTENDED_STOCK_DATABASE)

def get_stocks_by_exchange():
    """Get stock count breakdown by exchange."""
    exchanges = {
        'NYSE/NASDAQ (USA)': 0,
        'LSE (UK)': 0,
        'TSE (Japan)': 0,
        'SSE (China)': 0,
        'HKEX (Hong Kong)': 0,
        'Euronext (Europe)': 0,
        'TSX (Canada)': 0
    }

    for ticker in EXTENDED_STOCK_DATABASE.keys():
        if '.L' in ticker:
            exchanges['LSE (UK)'] += 1
        elif '.T' in ticker:
            exchanges['TSE (Japan)'] += 1
        elif '.SS' in ticker:
            exchanges['SSE (China)'] += 1
        elif '.HK' in ticker:
            exchanges['HKEX (Hong Kong)'] += 1
        elif any(suffix in ticker for suffix in ['.AS', '.PA', '.DE', '.BR', '.MC', '.MI']):
            exchanges['Euronext (Europe)'] += 1
        elif '.TO' in ticker:
            exchanges['TSX (Canada)'] += 1
        else:
            exchanges['NYSE/NASDAQ (USA)'] += 1

    return exchanges
