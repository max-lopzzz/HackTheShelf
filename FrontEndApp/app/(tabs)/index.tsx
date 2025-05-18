// app/(tabs)/index.tsx
import React, { useState } from 'react';
import { View, Text, StyleSheet, Image, Pressable, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';


export default function HomeScreen() {
  const [planogram, setPlanogram] = useState<string | null>(null);
  const [photo, setPhoto] = useState<string | null>(null);

 

const pickPlanogram = async () => {
  const result = await DocumentPicker.getDocumentAsync({
    type: 'image/*',
    copyToCacheDirectory: false,
  });

  // Modern discriminated union check
  if (result.assets && result.assets.length > 0) {
    const file = result.assets[0];
    setPlanogram(file.uri); // ✅ Now it's a string
  } else {
    // handle cancel or empty selection
    console.log('No document selected');
  }
};

  // take or pick photo
  const pickPhoto = async () => {
  const { granted } = await ImagePicker.requestCameraPermissionsAsync();
    if (!granted) return;

    // let the user take a photo first, otherwise fallback to library
    let result = await ImagePicker.launchCameraAsync({ quality: 0.5 });
    if (result.canceled) {
      result = await ImagePicker.launchImageLibraryAsync({ quality: 0.5 });
    }

    // new API returns { canceled: boolean, assets: [{ uri, ... }] }
    if (!result.canceled && result.assets.length > 0) {
      setPhoto(result.assets[0].uri);
    }
  };

  return (
    <View style={styles.container}>
      {/* Logo at top */}
      <Image source={require('@/assets/images/oxxo-logo.png')} style={styles.logo} />

      {/* Buttons row */}
      <View style={styles.row}>
        <Pressable
          style={({ pressed }) => [
            styles.square,
            pressed && styles.pressed,
            planogram && { backgroundColor: 'transparent' },
          ]}
          onPress={pickPlanogram}
        >
          {planogram
            ? <Image source={{ uri: planogram }} style={styles.thumb} />
            : <Text style={styles.plus}>＋</Text>
          }
        </Pressable>

        <Pressable
          style={({ pressed }) => [
            styles.square,
            pressed && styles.pressed,
            photo && { backgroundColor: 'transparent' },
          ]}
          onPress={pickPhoto}
        >
          {photo
            ? <Image source={{ uri: photo }} style={styles.thumb} />
            : <Text style={styles.camera}>📷</Text>
          }
        </Pressable>
      </View>

      {/* Compare button */}
      <Pressable style={({ pressed }) => [styles.compare, pressed && styles.pressed]} onPress={() => { /* TODO */ }}>
        <Text style={styles.compareText}>comparar</Text>
      </Pressable>
    </View>
  );
}

const SIZE = 120;
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    paddingTop: Platform.select({ ios: 60, android: 40 }),
  },
  logo: {
    width: 200,
    height: 80,
    resizeMode: 'contain',
    marginBottom: 40,
  },
  row: {
    flexDirection: 'row',
    gap: 20,
    marginBottom: 30,
  },
  square: {
    width: SIZE,
    height: SIZE,
    backgroundColor: '#F0B400',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  pressed: {
    opacity: 0.6,
  },
  plus: {
    fontSize: 48,
    color: '#FFF',
  },
  camera: {
    fontSize: 40,
  },
  thumb: {
    width: '100%',
    height: '100%',
    opacity: 0.4,
  },
  compare: {
    width: SIZE * 2 + 20,
    height: 60,
    backgroundColor: '#D32F2F',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  compareText: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: '600',
    textTransform: 'uppercase',
  },
});
