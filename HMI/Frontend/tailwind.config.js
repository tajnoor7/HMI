/** @type {import('tailwindcss').Config} */
const defaultTheme = require('tailwindcss/defaultTheme')
module.exports = {
  content: [
    "./src/**/*.{html,ts}",
  ],
  theme: {
    extend: {
      fontFamily: {
        poppins: ['"Poppins", sans-serif', ...defaultTheme.fontFamily.sans]
      },
      colors: {
        'white': '#fcfcfc',
        'black': '#141414',
        'gray': 'rgb(187 187 187)',
        'gray-light': 'rgb(187 187 187 / 36%)',
        'primary-color': '#2b85ff',
        'body-color': "#f0f5fb",
        'border-color': "#b5b5b5"

      },
    },
  },
  plugins: [],
}

