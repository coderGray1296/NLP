#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re

    ## { -CAPS, -INITC ap, -LC lowercase, 0 } +
    ## { -KNOWNLC, 0 } + [only for INITC]
    ## { -NUM, 0 } +
    ## { -DASH, 0 } +
    ## { -last lowered char(s) if known discriminating suffix, 0}

class getSignature():
    def __init__(self, word, language="es", lexSet=None):
        self.simple_unk_features = {}
        self.language = language
        sb = "UNK"
        isDigit = re.compile("\d")
        isLetter = re.compile("[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", re.UNICODE)
        isLowerCase = re.compile("[a-záéíóúüñ]", re.UNICODE)
        wlen = len(word)
        numCaps = 0
        hasDigit = False
        hasDash = False
        hasLower = False
        hasRefl = False
        hasVerb = False
        for i in range(wlen):
            char = word[i]
            if isDigit.search(char):
                hasDigit = True
            elif char == "-":
                hasDash = True
            elif isLetter.search(char):
                if isLowerCase.search(char):
                    hasLower = True
                else:
                    numCaps += 1
        char0 = word[0]
        lowered = word.lower()
        refl = lowered[-2:]
        if self.language == "es":
            if refl in ("se", "me", "te", "le", "la", "lo"):
                hasRefl = True
        if isLetter.search(char0) and not isLowerCase.search(char0):
            if numCaps == 1:
                sb += "_INITC"
                self.simple_unk_features["_INITC"] = {}
                #if lexSet and lowered in lexSet:
                #    sb += "_KNOWNLC"
            elif numCaps > 1:
                sb += "_CAPS"
                self.simple_unk_features["_INITC"] = {}
                self.simple_unk_features["_CAPS"] = {}
        # What is everything but the first letter is capitalized?
        elif isLetter.search(char0) and isLowerCase.search(char0):
             if numCaps == 1:
                 sb += "_NOTINITC"
                 self.simple_unk_features["_NOTINITC"] = {}
             elif numCaps > 1:
                sb += "_CAPS"
                self.simple_unk_features["_NOTINITC"] = {}
                self.simple_unk_features["_CAPS"] = {}
            # Oh well.  Berkeley seems to ignore this as well.
        elif numCaps > 1:
            sb += "_CAPS"
            self.simple_unk_features["_CAPS"] = {}
        else:
            sb += "_LC"
            self.simple_unk_features["_LC"] = {}
        if hasDigit:
            sb += "_NUM"
            self.simple_unk_features["_NUM"] = {}
        if hasDash:
            sb += "_DASH"
            self.simple_unk_features["_DASH"] = {}
        if wlen > 4 and not hasDash and not (hasDigit and numCaps > 0):
            #if tag != "v":
            #    continue
            # Spanish verb paradigms
            # Infinitive:
            # -ar, -ir, -er
            # present ar verbs:
            # -o, -as, -a, -amos, -áis, -an
            # present er/ir verbs:
            # -o, -es, -e, -emos, -imos, -éis, -en
            # progressive:
            # -ando, -iendo, -yendo
            # past ar:
            # -é, -aste, -ó, -amos, -asteis, -aron
            # past ir:
            # -í, -iste, -ió, -imos, -isteis, -ieron 
            # future (after ar, ir, er)
            # -é, -ás, á, emos, éis, án
            # imperfect ar:
            # -aba, -abas, -ábamas, abais, aban
            # imperfect ir:
            # -ía, ías, ía, íamos, íais, ían
            # Reflexive, etc:
            # -se, -me, -te, -le, -la, -lo
            # Perfect:
            # -ado, -ido, -ído, -to, -ierto, -uelto, -cho
            # Past subjunctive, ar:
            # -ara, -aras, -ara, -áramos, arais, aran
            # Past subjunctive, er/ir:
            # -iera, -ieras, iéramos, ierais, ieran
            # 'os' is not gonna be a verb, but whatever; it'll be common.
            if self.language == "es":
                endings = (u"iéramos", u"áramos", u"ábamas", u"íamos", u"isteis", u"ierais", u"asteis", u"íais", u"yendo", u"uelto", u"ierto", u"ieron", u"ieras", u"ieran", u"arais", u"abais", u"ído", u"ías", u"ían", u"éis", u"éis", u"áis", u"iste", u"imos", u"imos", u"iera", u"iedo", u"emos", u"aste", u"aron", u"aras", u"aran", u"ando", u"amos", u"amos", u"abas", u"aban", u"ía", u"ás", u"án", u"ió", u"ido", u"cho", u"ara", u"ado", u"aba", u"ó", u"í", u"é", u"á", u"to", u"ir", u"es", u"er", u"en", u"as", u"ar", u"an", u"os", u"o", u"e", u"a", u"s")
            elif self.language == "en":
                endings = ("ing", "ly", "y")
            for e in endings:
                if lowered.endswith(e):
                    sb += "_" + e
                    hasVerb = True
                    break
            if not hasVerb:
                if hasRefl:
                    for e in endings:
                        if lowered[:-2].endswith(e):
                            sb += "_" + e
                            hasVerb = True
                            break
                    sb += lowered[-2:]
        self.unk = unicode(sb)
        if hasVerb and len(e) > 2 and self.language == "es":
            self.simple_unk_features["_has_verb_ending"] = {}            
if __name__ == "__main__":
    examples = """El gobernante , con ganada fama desde que _ llegó hace 16 meses al poder de explotar al_máximo su oratoria y acusado por sus detractores de incontinencia verbal , enmudeció desde el momento en el que el Tribunal_Supremo_de_Justicia ( TSJ ) decidió suspender temporalmente los comicios múltiples ante la imposibilidad " técnica " de celebrarlos el 28_de_mayo .  Chávez se despidió del mundanal ruido el pasado_jueves con su más breve discurso por televisión , tildado de " institucional " por los observadores , en el que _ aceptó el aplazamiento de los comicios y _ valoró la " pedagógica " medida como un triunfo de la democracia venezolana .  Desde entonces _ entró en silencio absoluto .  Nadie sabe cuál es la nueva fecha que _ propone para las votaciones , ni si _ las quiere juntas o separadas , ni cuando _ va a reanudar la campaña .  Por su boca suelen hablar de_vez_en_cuando tanto el ministro de Relaciones_Exteriores , José_Vicente_Rangel , verdadero " portavoz " del Gobierno , como el presidente de la Comisión_Legislativa_Nacional ( CLN ) , o Congresillo , Luis_Miquilena , que es quien aparentemente tiene todos los resortes en el nuevo escenario electoral .  Por la Cancillería se sabe que la gira presidencial prevista entre el 16 y el 25_de_junio por varios países de la Organización_de_Países_Exportadores_de_Petróleo ( OPEP ) , con el fin de afinar detalles de_cara_a la II_Cumbre de ese cartel petrolero en Caracas , está en el alero hasta que se fijen los comicios .  Miquilena , mientras_tanto , se ha convertido una vez más desde que Chávez ocupa la Presidencia en el " comodín " de la partida , pues _ marcará la pauta sobre el inmediato futuro político .  Si el TSJ no lo remedia al admitir un recurso de la Defensoría_del_Pueblo para evitar que el Congresillo designe " a_dedo " mañana , viernes , a la nueva directiva del Consejo_Nacional_Electoral ( CNE ) , Miquilena nombrará a su criterio a las autoridades de ese organismo .  El CNE fue acusado hace unos días por la generalidad de las fuerzas políticas venezolanas de ser el verdadero y nico causante del aplazamiento de las votaciones debido_a la " incompetencia " de sus cinco miembros , que complicaron las ya de_por_sí complejas elecciones múltiples con continuas rectificaciones en las bases de datos del proceso automatizado .  El anterior CNE , elegido también por Miquilena , dimitió en pleno el pasado_lunes tras soportar una lluvia de críticas por su " falta de experiencia " y a sus cinco miembros se les ha prohibido la salida del país , mientras _ son investigados judicialmente por su comportamiento .  Ahora , el Congresillo quiere guardar las formas y consultar a la " sociedad civil " ( Iglesia , partidos , empresarios , sindicatos ) cuál sería la composición idónea y equilibrada del CNE , supuestamente garante de la imparcialidad de los comicios , con el fin de dar un tinte democrático a lo que será una designación .  Elías_Jaua , miembro del Congresillo , considera que los nuevos miembros del CNE deben tener experiencia para " dirigir procesos complejos " , y _ deben ser personas con independencia y equilibrio , respetabilidad y reconocida solvencia moral , algo que fue puesto en_entredicho con la anterior directiva .  Mientras , el ministro de Defensa , general Eliécer_Hurtado , se vio obligado a salir_al_paso de los ya casi normales y permanentes rumores de golpe de Estado , y _ negó que haya división en las filas castrenses y mucho menos que esté en marcha una conspiración contra Chávez .  " El pueblo puede estar seguro de que aquí no existe nada de conspiración , ni está preso nadie por esos motivos tampoco " , declaró Hurtado en una rueda de prensa .  _ Añadió que _ desconoce la procedencia de los rumores , que estarían motivados por una supuesta división en la milicia entre los partidarios de Chávez y los seguidores del aspirante presidencial y también militar retirado , Francisco_Arias_Cárdenas , quien sí se ha explayado al dar sus opiniones sobre la " payasada " del aplazamiento electoral  La amnistía favorece a los catorce coroneles detenidos y al más de un centenar de oficiales de menor rango procesados por participar en la asonada golpista contra Mahuad que facilitó la sucesión en la presidencia de Gustavo_Noboa .  Solicitada al Congreso por el propio jefe del Estado , el recurso de la amnistía ha sido exigida por todos los frentes sociales que han anunciado su incondicional apoyo a los militares rebeldes .  Noboa , que fue vicepresidente en el gobierno de Mahuad y le sucedió en el cargo tras su caída , considera que la amnistía permitirá la pacificación de la nación y la creación de un ambiente propicio para el diálogo y la concertación .  Y es que los coroneles rebeldes gozan de una amplia simpatía entre la población , pues , según las encuestas , la mayoría considera positivo el_que _ hayan apoyado a los movimientos sociales que exigían la salida de Mahuad , acusado de haber ahondado la crisis económica que afecta al país .  El cabecilla del movimiento militar fue el coronel Lucio_Gutiérrez , quien apoyó a los miles de indígenas que ocuparon el 21_de_enero el Palacio_Legislativo y luego marcharon hacia el centro de Quito para tomar posesión de la Casa_Presidencial .  Gutiérrez no se arrepiente de haber participado en la insurrección contra Mahuad y _ está seguro que la actitud de los oficiales se debió al elevado grado de corrupción que _ dice hubo durante la anterior administración .  El coronel quiere concluir su brillante carrera militar , aunque _ aún debe esperar las posibles sanciones disciplinarias que le podría imponer el mando militar .  La amnistía , según opiniones de diputados , no impide que las autoridades militares impongan sanciones disciplinarias a los oficiales involucrados , pues el recurso político sólo establece la suspensión de procesos penales civiles y los seguidos en la Corte_de_Justicia_Militar .  Los frentes populares no están dispuestos a que las Fuerzas_Armadas impongan sanciones y si , acaso , _ las aplican , _ convocarán a protestas contra los mandos militares , según lo aseguró el dirigente de la Coordinadora_de_Movimientos_Sociales , Napoleón_Saltos .  " _ Vamos a estar vigilantes de que se cumpla la Constitución y que la amnistía sea total , es que luego no se tomen retaliaciones o venganzas contra los militares a pretexto de una sanción disciplinaria " , indicó el dirigente .  _ Agregó que si los mandos militares no acatan a plenitud la amnistía , " que es una posición de la mayoría de la población " , e _ imponen sanciones disciplinarias que podrían llegar hasta la separación de los rebeldes de las filas castrenses , los grupos sociales retomarán las movilizaciones y se podría llegar " a una convulsión mayor " .  El presidente del Legislativo , Juan_José_Pons , coincidió en que la amnistía tiene por objetivo " pacificar al país " , y devolver la tranquilidad que la nación necesita para atender otros problemas urgentes como la crisis económica , la pobreza .  La aprobación de la amnistía " es una muestra de responsabilidad cívica del Congreso ecuatoriano , con el fin de " olvidarnos de los hechos pasados para construir una nueva historia " , afirmó Pons  Hace un par de días , en una nica acera _ vi cinco contenedores juntos : un contenedor metálico verde ( para reciclar vidrio ) ; un contenedor metálico amarillo ( para reciclar papeles : ¡ pero no cartones ! , _ aclaraba ) ; otro contenedor verde ( pero de plástico : para la basura normal ) ; otro contenedor amarillo ( más pequeño , acoplado al del papel , para reciclar pilas normales ) ; y otro más pequeño aún ( colocado sobre el anterior para las pilas botón ) .  Reciclaje , ahorro , aprovechamiento de los residuos y lucha contra el despilfarro energético de los países ricos .  Imbuido de ese espíritu , un escritor inglés - excepcional y sarcástico - proponía hace poco que no se tirase el agua de la bolsa de agua caliente .  Tirarla , _ decía , es un despilfarro ; lo que hay que hacer es reciclarla , guardarla y volverla a calentar al día siguiente .  Sí , señor .  Y , en la mesa , se acabó eso de usar los palillos una sola vez y tirarlos .  Hay que recogerlos una_vez usados , fregar los restos de comida que hayan quedado adheridos y ponerlos a secar .  De esta manera cada uno de nosotros evitará que hectáreas y hectáreas de bosque sean diezmadas cada día para fabricar esos palillos pequeñoburgueses con que _ hurgamos entre las caries .  Y , ya_que en eso _ estamos , otro consejo : los dientes cariados que nos arranque el dentista , nada_de tirarlos .  Convenientemente ensartados en un hilo de nylon , y alternados con los huesos de las aceitunas del aperitivo ( que _ habremos guardado , evidentemente : ¡ aquí no se tira nada ! ) _ conseguiremos un bonito collar recycled_ecological para regalar por_ejemplo a la asistenta .  El error fue haberlo comprado en rebajas .  Al terminar la Navidad pasada , una tienda de belenes ofreció nacimientos a precio de quema .  Y _ nos quemamos .  _ Hemos debido pensarlo .  _ Resultaban demasiado baratos para ser buenos .  Por eso , cuando _ destapamos la caja hace algunos días , _ descubrimos que los tres reyes eran blancos y que , en_vez_de un Niño_Jesús , había dos .  _ Eran idénticos .  La nica diferencia estaba en el color del pañal : el uno era blanco y el otro rojizo .  La ausencia de Baltasar era explicable pero alarmante : _ significaba que la ola racista acababa de llegar hasta los belenes .  Pero la existencia de dos recién nacidos en la misma caja sólo podía deberse a un descuido de fábrica .  De allí las rebajas .  - _ Ya sé qué ocurre - explicó una de mis hijas - . """
    words = examples.split()
    for word in words:
        sig = getSignature(word).unk
        print word, sig
