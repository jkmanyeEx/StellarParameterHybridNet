SELECT TOP 200
    sp.plate, sp.mjd, sp.fiberid,
    sp.teffadop, sp.loggadop, sp.fehadop
FROM sppParams AS sp
WHERE sp.teffadop BETWEEN 4000 AND 7000
  AND sp.loggadop BETWEEN 1.0 AND 5.0
  AND sp.fehadop BETWEEN -3.0 AND 0.5
  AND sp.fehadop > -5
  AND sp.loggadop > 0
  AND sp.seguePrimary = 1
  AND (sp.zwarning = 0 OR sp.zwarning = 16);
