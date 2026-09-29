# What the three tasks look like

Real inputs, real outputs from the fine-tuned model (`llama_3task`, dev split).
Contract text is excerpted around the relevant clause; quoted spans are verbatim.


## Task 1 — Compliance check (ContractNLI)

**Input:** the contract + one playbook position. **Output:** verdict + the language that justifies it.


### Entailment

**Playbook position:** Receiving Party may acquire information similar to Confidential Information from a third party.

**Contract (excerpt):**

```
… shall enjoy perpetual protection against unauthorized disclosure or use, subject only
to the provisions of the Trade Secrets Act and other applicable laws. 3. Limitations.
The term “Confidential Information” shall not include such portions of the Confidential
Information that: (i) are already published or available to the public other than as a
result of disclosure by the Receiving Party or any of its employees, officers,
directors, members, mana …
```

**Model output:**

```json
{
  "verdict": "Entailment",
  "evidence": [
    "The term \u201cConfidential Information\u201d shall not include such portions of the Confidential Information that:",
    "(ii) become available to the Receiving Party on a non-confidential basis from a source (other than the Disclosing Party or one of the Disclosing Party\u2019s employe\u2026"
  ]
}
```

### Contradiction

**Playbook position:** Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).

**Contract (excerpt):**

```
… Dealer without reference to the Confidential Information, or (vi) is required to be
disclosed by Dealer in compliance with applicable laws or court order. 2. Nondisclosure.
Dealer shall not distribute, disseminate, publish or disclose the Confidential
Information to any person, or entity not employed or retained by Dealer or the use of
the Confidential Information for any purpose other than to further the marketing and
sale of CIF products. Dealer will disclose the Confidential Information only to those
individuals employed by Dealer who need to know such Confidential Information. 3. Term.
The parties agree th …
```

**Model output:**

```json
{
  "verdict": "Contradiction",
  "evidence": [
    "Dealer shall not distribute, disseminate, publish or disclose the Confidential Information to any person, or entity not employed or retained by Dealer or the us\u2026"
  ]
}
```

### NotMentioned

**Playbook position:** Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.

**Contract (excerpt):**

```
COVENANT NOT TO COMPETE This Covenant Not to Compete (this "Agreement") is made this 3rd
day of April, 1996 by and among Lumex, Inc., a New York corporation ("Seller"), Lumex
Medical Products, Inc. (f/k/a MUL Acquisition Corp. I), a Delaware corporation
("Purchaser I"), MUL Acquisition Corp. II, a Delaware Corporation ("Purchaser II)
(Purchaser I and Purchaser II are collectively referred to herein as "Purchasers"), and
Fuqua Enterprises, Inc., a Delaware corporation ("Parent"). WITNESSETH WHEREAS, Seller
through …
```

**Model output:**

```json
{
  "verdict": "NotMentioned",
  "evidence": []
}
```


## Task 2 — Clause identification (CUAD)

**Input:** the contract + one clause category. **Output:** present or not, plus the clause text.


### Not detected — Non-Transferable License

**Contract (excerpt):**

```
EXHIBIT 'B' This Exhibit B is entered into as of the 26th day of March 2020 by and
between Kubient, Inc. ("Kubient"), and The Associated Press ("Customer"). This Exhibit
is hereby incorporated into and made a part of the Master Services Agreement (the
"Agreement") between the Parties (Effective Date: February 5, 2020). NATURE OF
ENGAGEMENT: Customer has retained Kubient to help increase revenue from its consumer
offerings, including but not limited to its websites, apps, videos, and podcasts.
SERVICES: Kubient …
```

**Model output:**

```json
{
  "present": false,
  "evidence": []
}
```

### Found — Anti-Assignment

**Contract (excerpt):**

```
… Q, at PCQ's option and expense, to control the defense,      settlement or other
handling of such claim.  10. Assignment  Source: PCQUOTE COM INC, S-1/A, 7/21/1999
This Agreement or any rights or obligations granted hereunder may not be assigned by ABW
without the prior written consent of PCQ.  11. Applicable Law and Venue      This
Agreement shall be interpreted, construed and enforced in all respects in accordance
with the laws of the State of Illinois, except …
```

**Model output:**

```json
{
  "present": true,
  "evidence": [
    "This Agreement or any rights or obligations granted hereunder may not be assigned by ABW without the prior written consent of PCQ."
  ]
}
```


## Task 3 — Risk note (synthetic)

**Input:** one clause + its category. **Output:** one plain-English sentence.


### License Grant

**Clause:**

```
In addition, to the extent that it is legally able to do so, MD Anderson hereby grants
LBIO a royalty-free right and license to use and reproduce any Publication. MD Anderson
hereby grants LBIO a non-exclusive, royalty free, perpetual license (with rights to sub-
license) under, in and to all Background Intellectual Property that is: (a) owned by MD
Anderson; (b) consists of and/or comprises the manufacturing protocol utilized by MD
Anderson in the conduct of a Study; and (c) reasonably necessary to exploit (including
developing, obtaining and maintaining
```

**Model output:** If MD Anderson owns or uses certain intellectual property or data related to TILs in double refractory melanoma, it gives LBIO the right to use and share that property without paying royalties, as long as it's not already owned by a third party.


### Insurance

**Clause:**

```
Commercial General Liability Insurance providing the limits of coverage written on an
occurrence basis, in no event less than $1,000,000 combined single limit and $2,000,000
in the aggregate for personal and bodily injury and death arising therefrom and Broad
Form property damage arising out of any one occurrence in connection with the Work or
any part thereof, which insurance shall include coverage with the same minimum limits
for contractual liability and completed operations liability. Motor Vehicle Insurance
covering bodily injury, death of a person
```

**Model output:** This clause requires you to provide and maintain specific insurance coverage for the work, including liability and property damage, and to name EDGE and Mediacom as additional insureds, which becomes effective when the work begins and must be kept in force throughout the agreement.
