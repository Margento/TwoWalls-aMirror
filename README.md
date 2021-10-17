# TwoWalls&aMirror
Transmedia poem commissioned by New York Romanian Cultural Institute (ICR NY), to be released on Oct 19th 2 pm NYC time on the institue's Facebook page. #PathFinding #Multilingual #CorpusAlternating #VectorProsody #GraphPoem


[If you want to run the code in the Python (.py) scripts--which is in fact the same as the one in the Jupyter Notebooks, it is just that the latter also include the outputs--organize your folders the way they are in this repo and also save and extract the FastText word embeddings for the languages you need (choose the text-format files at https://fasttext.cc/docs/en/crawl-vectors.html) into the root one ("/fastText_multilingual-master/" in this particular case).] 



Margento 

Two Walls and a Mirror

—Poetics Statement— 

The world of the pandemic has been a 2-dimensional one. Even our cubicles have been crushed flat by the lockdown. The only beyond was (is?) the digital one—interfaces and web-based algorithms—which worked in fact like a mirror. A mirror of our own discontent that every now and then seemed to work like a two-way mirror (and two-way screen) of ubiquitous control and surveillance. Yet what else is there for us to see besides the reflection of the two walls enclosing us and thus turning the mirror in its turn into a third hypnotizing wall? 

This work consists of two symmetrical bilingual (English and Romanian) components that are in and of themselves instances of hybridization, recycling, sampling, and e-scape. 

The main recycled part comes from the book Various Wanted / Se caută diversuri (by Margento, Steve Rushton, and Taner Murat, 2021, https://bit.ly/2WS7tZl) which involved itself recycling a number of English ‘translations’ of Ovid into further literary, computational, and visual translations. I specifically used the section titled “Festive Feminization,” a computational (topic-modeling-based) translation of an Ovidian ‘translational original’ by Steve Rushton which I hybridized with dozens of poems, academic articles, and social media content by women authors and artists. 

The first component thus consisted of an English wall, “Festive Feminization,” and a Romanian one, the pandemie poezie reciclare hibrid (without quotes) Google search results. I kept searching on Google till I got the same number of ‘satisfying’ results as the sections of “Festive Feminization” and edited them as ‘poems’ [selecting only what appealed to me and rearranging the resulting text in verse].

I translated the bilingual corpus into a network by using the multilingual FastText vectors and then I kept only the English-Romanian edges to try and render the way in which life in lockdown was mostly about ‘flat’ binary alternatives or even walking in circles.

There was therefore navigation on search engines and across corpora, a conjoined form of e-scape involving both e-space flânerie and route-finding (more precisely, bridge problem) algorithmics. My compositional principle was to find the route progressing from the strongest English-Romanian correlation in the network to the weakest one (while one can of course choose to go the other way round as well) and generating a poem that best describes that cross-lingual e-scape. I needed the route not to hit any ‘spot’ (poem-node) more than once—the foundational Euler walk problem, yet without the cross-ALL-bridges-in-Koenigsberg constraint—and the journey to reach its end the moment all neighbors of a poem across the language barrier have already been visited.

Once that was accomplished, I needed my algorithms to output a poem, one conciser than the sum of all of the poems on the route. That took me to a machine-learning-based poetic form I had previously explored: vector prosody (see #GraphPoem @ DHSI 2021 on YouTube—https://bit.ly/3jML9ZW—and my forthcoming article in IDEAH issue 2). I selected therefore the line in each poem whose vector was the closest (in terms of cosine similarity) to the one of the respective poem as a whole. That sequence of alternating English and Romanian lines was my poem. 

My first poem, that is, of a potentially countless series of poems obtained by changing or expanding the input and/or the method specifics. 

The second component of this work then consisted, symmetrically, of the corresponding Romanian section of Various Wanted / Se caută diversuri, “Zaiafet - feminizare,” combined with the results of the pandemic poetry recycle hybrid Google search (no quotes again). I similarly found a cross-lingual alternating path and thus obtained a representative Romanian and English poem documenting the journey of going back and forth between two languages, two corpora, two walls, the two sides of a two-way mirror reflecting two walls…

I circled—or rather… squared—back to my lockdown mood/medium when I translated the English-Romanian-path poem into English and the Romanian-English-path one into Romanian, and thus got four poems, four walls erected around me. And the mirror in between, my spot, the sweeping code-translational diagonal. 



Margento

Două ziduri și o oglindă

- Despre poetica proiectului - 

Lumea pandemiei a fost una bidimensională. Chiar și cutiuțele în care locuiam sau lucram au fost aplatizate de izolare. Singurul dincolo a fost (este?) digitalul - interfețe, algoritmi de web - funcționând, de fapt, ca o oglindă. Oglinda vrajbei noastre care, din când în când, părea să funcționeze ca o oglindă bidirecțională (și ecran bilateral) a controlului și supravegherii omniprezente. Și totuși, ce altceva puteam vedea acolo în afară de reflecția celor doi pereți care ne înconjoară, transformând astfel oglinda, la rândul ei, într-un al treilea zid hipnotizant? 

Lucrarea de față constă din două componente bilingve simetrice (engleză și română) care sunt ele însele cazuri de hibridizare, reciclare, eșantionare și e-(vizio)migrare. 

Principala parte reciclată provine din volumul Various Wanted / Se caută diversuri (Margento, Steve Rushton și Taner Murat, 2021, https://bit.ly/2WS7tZl) care a implicat ea însăși reciclarea unui număr de "traduceri" englezești ale lui Ovidiu pentru obținerea altor traduceri literare, computaționale și vizuale. Am folosit pentru asta secțiunea intitulată “Zaiafet - feminizare“, o traducere computațională (bazată pe modelare de subiecte) a unui "original traductologic" ovidian aparținând lui Steve Rushton, pe care am hibridizat-o cu zeci de poezii, articole academice, postări în rețelele sociale, etc., ale unor femei scriitoare, universitare sau artiste. 

Astfel, prima componentă a constat dintr-un perete în română, "Zaiafet - feminizare", și unul în engleză, rezultatele căutării pe Google pandemic poetry recycle hybrid (fără ghilimele). Am continuat să caut pe Google până când am obținut același număr de rezultate "satisfăcătoare" ca și secțiunile din "Zaiafet - feminizare" și le-am editat ca "poeme" [selectând doar ceea ce mi s-a părut valabil și rearanjând textul rezultat în versuri].

Am tradus corpus-ul bilingv într-o rețea folosind vectorii multilingvi FastText și apoi am păstrat doar muchiile româno-englezești pentru a încerca să redau modul în care viața în izolare era în mare parte despre alternative binare, “plate", sau chiar despre mersul în cerc.

Prin urmare, a existat o navigare pe motoarele de căutare și prin corpusuri, o formă contopită de e-(vizio)migrare implicând atât hoinăreala în e-spațiu, cât și algoritmica găsirii de rute (într-un sens foarte înrudit cu "problema podurilor” din matematică). Principiul meu de compoziție a fost acela de a găsi traseul care să avanseze gradat de la corelația româno-engleză cea mai puternică din rețea la cea mai slabă (deși se poate alege, desigur, și progresia inversă) și de a genera un poem care să descrie cel mai bine această e-(vizio)migrare multilingvă. Aveam nevoie ca traseul să nu atingă niciun "punct" (nod-poem) mai mult de o dată - problema fundamentală a celor șapte poduri, a lui Euler, dar fără constrângerea de a traversa TOATE podurile din Königsberg - și ca această călătorie să se încheie în momentul în care toți vecinii de altă limbă ai unui poem au fost deja vizitați.

După ce acest lucru a fost realizat, aveam nevoie ca algoritmii mei să producă un poem, unul mai concis decât suma tuturor poemelor de pe traseu. Asta m-a dus la o formă poetică pe care o explorasem anterior: prozodia vectorială (bazată pe învățare automată - machine learning - cf. #GraphPoem @ DHSI 2021 pe YouTube - https://bit.ly/3jML9ZW - și articolul ce urmează să apară pe această temă în numărul 2 al revistei IDEAH). Am selectat, așadar, versul din fiecare poem al cărui vector era cel mai apropiat (din punct de vedere al similarității de tip cosinus) de cel al poemului respectiv în ansamblu. Acea secvență de versuri alternativ românești și englezești a reprezentat poemul meu. 

Primul, de fapt, dintr-o serie potențial nenumărată de poeme obținute prin schimbarea sau mărirea mulțimii datelor de intrare și/sau a diverselor detalii de metodă. 

Cea de-a doua componentă a acestei lucrări a constat apoi, simetric, din secțiunea engleză corespunzătoare din Various Wanted / Se caută diversuri, “Festive Feminization", combinată cu rezultatele căutării pandemie poezie reciclare hibrid pe Google (din nou fără ghilimele). Am găsit în mod similar o cale de alternanță interlingvistică și am obținut astfel un poem reprezentativ în română și engleză care documentează călătoria de a merge în zig-zag între două limbi, două corpusuri, doi pereți, cele două fețe ale unei oglinzi bidirecționale reflectând cei doi pereți, etc.

Printr-o completă mișcare circulară - sau mai degrabă... rectangulară - am revenit astfel la starea/mediul meu de izolare când am tradus poemul alcătuit pe calea anglo-română în engleză și poemul produs de ruta româno-engleză în română și am obținut astfel patru poeme, patru ziduri ridicate în jurul meu. Și totodată, oglinda dintre ele, locul meu, diagonala de programare-traducere rotitoare.
