
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

from PIL import Image
from streamlit_drawable_canvas import st_canvas


# CONFIGURACION

st.set_page_config(
    page_title="TecnoForms - Reconocimiento de Digitos",
    page_icon="🔢",
    layout="centered"
)


# CARGAR MODELO

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelos/modelo_mnist.keras")


model = cargar_modelo()


# PROCESAMIENTO DE IMAGENES

def procesar_imagen(imagen):

    if imagen.shape[2] == 4:
        gris = cv2.cvtColor(imagen, cv2.COLOR_RGBA2GRAY)
    else:
        gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)

    _, binaria = cv2.threshold(
        gris, 180, 255, cv2.THRESH_BINARY_INV
    )

    contornos, _ = cv2.findContours(
        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    digitos = []

    for contorno in contornos:
        if cv2.contourArea(contorno) >= 20:
            digitos.append(cv2.boundingRect(contorno))

    digitos.sort(key=lambda d: d[0])

    if not digitos:
        return None, None, "No se detectaron digitos."

    if len(digitos) > 9:
        return None, None, "Se detectaron mas de 9 digitos."

    datos, visuales = [], []

    for x, y, w, h in digitos:

        recorte = Image.fromarray(binaria[y:y+h, x:x+w])
        recorte.thumbnail((20, 20), Image.Resampling.LANCZOS)

        final = Image.new("L", (28, 28), 0)
        final.paste(
            recorte,
            ((28 - recorte.width) // 2,
             (28 - recorte.height) // 2)
        )

        datos.append(np.array(final, dtype="float32") / 255.0)
        visuales.append(final)

    datos = np.array(datos).reshape(-1, 28, 28, 1)

    return datos, visuales, None


# RECONOCIMIENTO

def reconocer(imagen):

    datos, visuales, error = procesar_imagen(imagen)

    if error:
        st.warning(error)
        return

    predicciones = model.predict(datos, verbose=0)
    numeros = np.argmax(predicciones, axis=1)
    confianzas = np.max(predicciones, axis=1) * 100

    resultado = "".join(map(str, numeros))

    st.success("Analisis completado")

    st.markdown("### Resultado del reconocimiento")

    st.markdown(
        f"<h1 style='text-align:center;color:#16a34a;"
        f"font-size:55px'>{resultado}</h1>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    col1.metric("Digitos reconocidos", len(numeros))
    col2.metric("Confianza promedio", f"{np.mean(confianzas):.2f}%")

    st.markdown("### Reconocimiento individual")

    columnas = st.columns(len(numeros))

    for i, columna in enumerate(columnas):
        with columna:
            st.image(visuales[i], width=55)
            st.markdown(f"**{numeros[i]}**")
            st.caption(f"{confianzas[i]:.1f}%")


# INTERFAZ PRINCIPAL

st.title("🔢 TecnoForms")
st.subheader("Reconocimiento de numeros manuscritos")

st.write("Selecciona como deseas ingresar los numeros.")

opcion = st.radio(
    "Metodo de reconocimiento:",
    ["Dibujar numeros", "Subir una imagen"],
    horizontal=True
)


# DIBUJAR NUMEROS

if opcion == "Dibujar numeros":

    st.info("Dibuja hasta 9 digitos procurando que no se toquen.")

    canvas = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=12,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=180,
        width=630,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("🔍 Reconocer numero", use_container_width=True):

        if canvas.image_data is not None:
            reconocer(canvas.image_data.astype("uint8"))


# SUBIR IMAGEN

else:

    st.info("Sube una imagen que contenga numeros manuscritos.")

    archivo = st.file_uploader(
        "Selecciona una imagen",
        type=["png", "jpg", "jpeg"]
    )

    if archivo:

        imagen = Image.open(archivo).convert("RGB")

        st.image(
            imagen,
            caption="Imagen cargada",
            use_container_width=True
        )

        if st.button("🔍 Reconocer numeros", use_container_width=True):
            reconocer(np.array(imagen))


# PIE DE PAGINA

st.divider()

st.caption("TecnoForms | Sistema de reconocimiento de numeros manuscritos")
st.caption("Modelo desarrollado mediante redes neuronales convolucionales (CNN).")

'''
Para que corra la simulacion insertar en la terminal:
python -m streamlit run app.py
'''
 